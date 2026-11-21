// Package hesgpu runs the holonomy–eikonal field's sweeps on the GPU through
// dawnrt: the fast-sweeping / Bellman–Ford form the HES note names for
// parallel hardware and training, where the MRH writer (serial, CPU) is the
// inference path. Forward sweeps give the hard field or the soft-min field at
// temperature 1/β; adjoint sweeps give the exact gradient of a loss on that
// field with respect to every edge weight and source birth (hes.SoftGrad is
// the same computation on the CPU).
package hesgpu

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"sync/atomic"

	"gptrun/internal/hes"
	"gptrun/internal/rt"
)

//go:embed sweep.wgsl
var shader string

// inf is the kernel's +∞.
const inf = float32(3.0e38)

var seq atomic.Int64

// Sweeper holds one graph and metric on the device.
type Sweeper struct {
	d    *rt.Dev
	own  bool
	g    *hes.Graph
	w    []int
	n    int
	pre  string
	ids  map[string]uint32
	fwd  uint32
	adj  uint32
	grp  map[string]uint32
	// Batch is how many sweeps go in one submit between convergence checks.
	Batch int
}

// Open puts g with edge weights w (nil = base) on d. d nil starts dawnrt.
func Open(d *rt.Dev, g *hes.Graph, w []int) (*Sweeper, error) {
	own := false
	if d == nil {
		var err error
		if d, err = rt.Open(); err != nil {
			return nil, err
		}
		own = true
	}
	if w == nil {
		w = g.W
	}
	if len(w) != len(g.Edges) {
		return nil, fmt.Errorf("hesgpu: %d weights for %d edges", len(w), len(g.Edges))
	}
	s := &Sweeper{d: d, own: own, g: g, w: w, n: g.N, pre: fmt.Sprintf("hes%d.", seq.Add(1)),
		ids: map[string]uint32{}, grp: map[string]uint32{}, Batch: 32}
	if err := s.boot(); err != nil {
		s.Close()
		return nil, err
	}
	return s, nil
}

func (s *Sweeper) name(x string) string { return s.pre + x }

func u32s(v []int) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
	}
	return b
}

func f32s(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}

func readF32(b []byte, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

func (s *Sweeper) boot() error {
	d := s.d
	off, nbr, eid := s.g.CSR()
	wa := make([]float32, len(nbr))
	for i, e := range eid {
		wa[i] = float32(s.w[e])
	}
	nb := 4 * s.n
	bufs := []struct {
		n    string
		data []byte
		size int
	}{
		{"prm", nil, 16}, {"off", u32s(off), 0}, {"nbr", u32s(nbr), 0}, {"w", f32s(wa), 0},
		{"s", nil, nb}, {"a", nil, nb}, {"b", nil, nb}, {"dl", nil, nb}, {"tf", nil, nb},
	}
	for _, b := range bufs {
		id, err := d.Buf(s.name(b.n), b.data, b.size)
		if err != nil {
			return err
		}
		s.ids[b.n] = id
	}
	if err := d.Shader(s.name("sweep"), shader); err != nil {
		return err
	}
	ro, rw, uni := byte(rt.KindRO), byte(rt.KindRW), byte(rt.KindUni)
	var err error
	if s.fwd, err = d.Pipe(s.name("fwd"), s.name("sweep"), "forward", "", []byte{uni, ro, ro, ro, ro, ro, rw, rw}); err != nil {
		return err
	}
	if s.adj, err = d.Pipe(s.name("adj"), s.name("sweep"), "adjoint", "", []byte{uni, ro, ro, ro, ro, ro, rw, rw, ro}); err != nil {
		return err
	}
	group := func(gname, pipe, in, out string, tf bool) error {
		binds := []rt.Bind{{Buf: s.name("prm"), Size: 16}, {Buf: s.name("off")}, {Buf: s.name("nbr")}, {Buf: s.name("w")},
			{Buf: s.name("s")}, {Buf: s.name(in)}, {Buf: s.name(out)}, {Buf: s.name("dl")}}
		if tf {
			binds = append(binds, rt.Bind{Buf: s.name("tf")})
		}
		id, err := d.Group(s.name(gname), s.name(pipe), binds)
		if err != nil {
			return err
		}
		s.grp[gname] = id
		return nil
	}
	for _, g := range []struct {
		n, p, in, out string
		tf            bool
	}{{"fab", "fwd", "a", "b", false}, {"fba", "fwd", "b", "a", false}, {"aab", "adj", "a", "b", true}, {"aba", "adj", "b", "a", true}} {
		if err := group(g.n, g.p, g.in, g.out, g.tf); err != nil {
			return err
		}
	}
	return nil
}

// Close releases a dawnrt this sweeper started.
func (s *Sweeper) Close() {
	if s.own && s.d != nil {
		s.d.Close()
	}
}

func (s *Sweeper) prm(beta float64, hard bool) []byte {
	b := make([]byte, 16)
	binary.LittleEndian.PutUint32(b[0:], uint32(s.n))
	if hard {
		binary.LittleEndian.PutUint32(b[4:], 1)
	}
	binary.LittleEndian.PutUint32(b[8:], math.Float32bits(float32(beta)))
	return b
}

// run sweeps from the vector already in "a" with pipeline pipe and groups
// (ab, ba) until the largest change is below tol or max sweeps, and returns
// the result and the last change.
func (s *Sweeper) run(writes []rt.Write, pipe uint32, ab, ba string, max int, tol float64) ([]float32, int, float64, error) {
	x := uint32((s.n + 255) / 256)
	batch := s.Batch
	if batch < 2 {
		batch = 2
	}
	batch &^= 1 // even: the result lands back in "a"
	done := 0
	for {
		var ops []rt.Op
		for i := 0; i < batch; i += 2 {
			ops = append(ops,
				rt.Op{Tag: rt.OpDispatch, Pipe: pipe, Group: s.grp[ab], X: x, Y: 1, Z: 1},
				rt.Op{Tag: rt.OpDispatch, Pipe: pipe, Group: s.grp[ba], X: x, Y: 1, Z: 1})
		}
		out, err := s.d.Run2(writes, ops, []rt.Get{{Buf: s.ids["a"], N: 4 * s.n}, {Buf: s.ids["dl"], N: 4 * s.n}}, true)
		if err != nil {
			return nil, done, 0, err
		}
		writes = nil
		done += batch
		res := readF32(out, s.n)
		dl := readF32(out[4*s.n:], s.n)
		var delta float64
		for _, v := range dl {
			delta = math.Max(delta, float64(v))
		}
		if delta < tol || done >= max {
			return res, done, delta, nil
		}
	}
}

// Arrival is the field of the given sources: soft-min at β, or with hard the
// plain min, T(v) = min_i(b_i + d(s_i, v)).
func (s *Sweeper) Arrival(src []int, birth []float64, beta float64, hard bool, maxSweeps int, tol float64) (hes.SoftField, error) {
	S := make([]float64, s.n)
	for v := range S {
		S[v] = math.Inf(1)
	}
	for i, v := range src {
		if math.IsInf(S[v], 1) {
			S[v] = birth[i]
		} else if hard {
			S[v] = math.Min(S[v], birth[i])
		} else {
			m := math.Min(S[v], birth[i])
			S[v] = m - math.Log(math.Exp(-beta*(S[v]-m))+math.Exp(-beta*(birth[i]-m)))/beta
		}
	}
	s32 := make([]float32, s.n)
	for v, x := range S {
		s32[v] = inf
		if !math.IsInf(x, 1) {
			s32[v] = float32(x)
		}
	}
	sb := f32s(s32)
	writes := []rt.Write{{Buf: s.ids["prm"], Data: s.prm(beta, hard)}, {Buf: s.ids["s"], Data: sb}, {Buf: s.ids["a"], Data: sb}}
	res, sweeps, delta, err := s.run(writes, s.fwd, "fab", "fba", maxSweeps, tol)
	if err != nil {
		return hes.SoftField{}, err
	}
	T := make([]float64, s.n)
	for v, x := range res {
		T[v] = math.Inf(1)
		if x < inf {
			T[v] = float64(x)
		}
	}
	return hes.SoftField{Beta: beta, T: T, S: S, Sweeps: sweeps, Delta: delta}, nil
}

// Grad is the exact gradient of L = Σ_v dL[v]·T(v) at the soft field sf,
// with respect to every edge weight and every vertex's source birth.
func (s *Sweeper) Grad(sf hes.SoftField, dL []float64, maxSweeps int, tol float64) (dW, dS []float64, err error) {
	t32 := make([]float32, s.n)
	for v, x := range sf.T {
		t32[v] = inf
		if !math.IsInf(x, 1) {
			t32[v] = float32(x)
		}
	}
	d32 := make([]float32, s.n)
	for v, x := range dL {
		d32[v] = float32(x)
	}
	db := f32s(d32)
	writes := []rt.Write{{Buf: s.ids["prm"], Data: s.prm(sf.Beta, false)}, {Buf: s.ids["s"], Data: db},
		{Buf: s.ids["a"], Data: db}, {Buf: s.ids["tf"], Data: f32s(t32)}}
	mu32, _, _, err := s.run(writes, s.adj, "aab", "aba", maxSweeps, tol)
	if err != nil {
		return nil, nil, err
	}
	beta, T, S := sf.Beta, sf.T, sf.S
	share := func(from, to float64) float64 {
		if math.IsInf(from, 1) || math.IsInf(to, 1) {
			return 0
		}
		return math.Exp(-beta * (from - to))
	}
	dW = make([]float64, len(s.g.Edges))
	for e, uv := range s.g.Edges {
		u, v := uv[0], uv[1]
		we := float64(s.w[e])
		dW[e] = float64(mu32[v])*share(T[u]+we, T[v]) + float64(mu32[u])*share(T[v]+we, T[u])
	}
	dS = make([]float64, s.n)
	for v := range dS {
		dS[v] = float64(mu32[v]) * share(S[v], T[v])
	}
	return dW, dS, nil
}
