package hes

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// HES is the whole holonomy–eikonal state of the design note: a fixed-size
// memory with no token buffer, built from two associative path sums.
//
// Channel 1 (holonomy) is the running product g_t = g_{t−1}·T(x_t) in S_n
// (perm.go) and, beside it, in SO(n) by Householder pairs (holonomy.go).
//
// Channel 2 (eikonal) is the field on the latent torus K (graph.go, eikonal.go).
// A write maps its key to a source vertex s_t = φ(x_t) (keymap.go), takes the
// speed field of its token class F_t = f(x_t) (speed.go), enters the queue at
// κ = µ + c(x_t) with salience c from how much of the write is new, carries
// the recency head start b = −λ·i, and marches on the multi-radix heap until
// the front stops improving. Recency, similarity under the metric and
// salience compete in that one min: the Johnson–Mehl overwrite rule.
//
// A read maps the query to a vertex and follows −∇T back along the
// characteristic to the basin, returning the payload and H there.
//
// λ comes from the retention law's design rule for a target window
// (laws.go), the metric scale may follow a schedule with re-anchoring
// (Theorem 3), and the persistent H₁ of {T ≤ τ} with the holonomy around each
// cycle is the obstruction readout (persist.go). Save and Load keep the state
// across runs: N never grows, so the file has a fixed size per vertex.
type HES struct {
	Opt Options
	G   *Graph
	F   *Eik
	Sp  *Speeds
	K   *KeyMap

	g       Perm
	rot     Rot
	nWrite  int
	lambda  int
	seen    map[int]bool
	rotOf   []Rot // per payload id: the SO(n) holonomy when it was written
	classOf []int // per payload id: its speed class
	growth  *Growth
	anchor  *Anchor
}

// Options sizes one state. Zero values take the defaults in NewHES.
type Options struct {
	Dims     []int   // torus sides; default 64×64 (N = 4096)
	Weight   int     // base edge weight; default 1000
	Jitter   float64 // edge weights within ±Jitter; default 0.2
	Seed     int64   // graph, key map and speed fields
	Rank     int     // S_n degree; default 5
	RotN     int     // SO(n) channel size; default 8, 1 turns it off
	Lambda   int     // head start per write in weight units; −1: design rule from Window
	Window   int     // retention target W for the design rule
	Salience int     // the largest salience offset c; default one base edge, −1 none
	Classes  int     // token classes of the speed field; default 8, 1 is the base metric
	Kappa    float64 // speed multipliers within (1/κ, κ); default 2
	Sigma    float64 // speed field smoothing in cells; default 4
	MStar    float64 // re-anchoring threshold; default 2.08
	NoLearn  bool    // keep φ the fixed projection (no landmarks)
}

func (o *Options) fill() {
	if len(o.Dims) == 0 {
		o.Dims = []int{64, 64}
	}
	if o.Weight == 0 {
		o.Weight = 1000
	}
	if o.Jitter == 0 {
		o.Jitter = 0.2
	}
	if o.Rank == 0 {
		o.Rank = S5
	}
	if o.RotN == 0 {
		o.RotN = 8
	}
	if o.Salience == 0 {
		o.Salience = o.Weight
	}
	if o.Classes == 0 {
		o.Classes = 8
	}
	if o.Kappa == 0 {
		o.Kappa = 2
	}
	if o.Sigma == 0 {
		o.Sigma = 4
	}
	if o.MStar == 0 {
		o.MStar = 2.08
	}
}

// NewHES builds an empty state.
func NewHES(o Options) (*HES, error) {
	o.fill()
	g, err := Torus(o.Dims, o.Weight, o.Jitter, o.Seed)
	if err != nil {
		return nil, err
	}
	f, err := NewEik(g, o.Rank)
	if err != nil {
		return nil, err
	}
	h := &HES{Opt: o, G: g, F: f, Sp: NewSpeeds(g, o.Classes, o.Kappa, o.Sigma, o.Seed+7),
		g: Identity(o.Rank), rot: RotIdentity(max(o.RotN, 1)), seen: map[int]bool{}}
	h.lambda = max(o.Lambda, 0)
	if o.Lambda < 0 && o.Window > 0 {
		h.lambda = h.designLambda(o.Window)
	}
	h.anchor = NewAnchor(float64(h.lambda), 1, o.MStar)
	return h, nil
}

// Lambda is the head start per write in use.
func (h *HES) Lambda() int { return h.lambda }

// Writes is the number of writes so far.
func (h *HES) Writes() int { return h.nWrite }

// Holonomy is the S_n channel g_t.
func (h *HES) Holonomy() Perm { return append(Perm(nil), h.g...) }

// Rotation is the SO(n) channel.
func (h *HES) Rotation() Rot { return h.rot.Clone() }

// Growth measures (once) the ball growth of the base metric.
func (h *HES) Growth() Growth {
	if h.growth == nil {
		gr := MeasureGrowth(h.G, nil, 32, h.Opt.Seed+11)
		h.growth = &gr
	}
	return *h.growth
}

func (h *HES) designLambda(w int) int {
	gr := h.Growth()
	lam := DesignLambda(h.G.N, w, gr.C, gr.D)
	if lam <= 0 {
		return 0
	}
	if !DesignWindow(lam, w, gr.WMin, gr.Diam, gr.D) {
		lam = DesignLambdaExact(h.tables(16), w, 0, float64(gr.Diam))
	}
	return max(1, int(math.Round(lam)))
}

func (h *HES) tables(n int) []Ball {
	rng := rand.New(rand.NewSource(h.Opt.Seed + 13))
	out := make([]Ball, n)
	for i := range out {
		out[i] = NewBall(h.G, rng.Intn(h.G.N), nil, nil)
	}
	return out
}

// ensureKeys makes the key map at the first key's size.
func (h *HES) ensureKeys(m int) error {
	if h.K != nil {
		if h.K.M != m {
			return fmt.Errorf("hes: key of %d values, the map takes %d", m, h.K.M)
		}
		return nil
	}
	k, err := NewKeyMap(m, h.G.Dim(), h.Sp.K, h.Opt.RotN, h.Opt.Seed+3)
	if err != nil {
		return err
	}
	h.K = k
	return nil
}

// FitKeys fits φ to sample keys: their leading principal directions, then the
// spread. With writes already made the map only recalibrates if it was never
// used, since moving φ would orphan the stored sources.
func (h *HES) FitKeys(keys [][]float32) error {
	if len(keys) == 0 {
		return nil
	}
	if h.nWrite > 0 && h.K != nil {
		return fmt.Errorf("hes: φ is fixed once writes are stored")
	}
	if err := h.ensureKeys(len(keys[0])); err != nil {
		return err
	}
	h.K.Fit(keys)
	return nil
}

// CalibrateKeys sets φ's spread from sample keys, keeping its directions.
func (h *HES) CalibrateKeys(keys [][]float32) error {
	if len(keys) == 0 || (h.nWrite > 0 && h.K != nil) {
		return nil
	}
	if err := h.ensureKeys(len(keys[0])); err != nil {
		return err
	}
	h.K.Calibrate(keys)
	return nil
}

// Item is one write: the token ids it develops, its key vector (nil maps by
// the ids' hash), and the payload a read returns.
type Item struct {
	IDs  []int
	Key  []float32
	Text string
}

// WriteInfo reports one write.
type WriteInfo struct {
	Vertex, Class, Birth, Salience, Took int
	Hol                                  Perm
}

// Vertex is φ for a key or, without one, for the ids.
func (h *HES) Vertex(key []float32, ids []int) (int, error) {
	if len(key) > 0 {
		if err := h.ensureKeys(len(key)); err != nil {
			return 0, err
		}
		return h.K.Vertex(h.G, key), nil
	}
	return phi(ids, h.G.N), nil
}

// Write develops both holonomy channels by the item and marches its source.
func (h *HES) Write(it Item) (WriteInfo, error) {
	var info WriteInfo
	for _, id := range it.IDs {
		h.g = h.g.Mul(FromToken(h.Opt.Rank, id))
	}
	if len(it.Key) > 0 {
		if err := h.ensureKeys(len(it.Key)); err != nil {
			return info, err
		}
		if h.Opt.RotN > 1 {
			h.rot = h.rot.Mul(h.K.Householder(it.Key))
		}
	}
	info.Hol = h.Holonomy()
	if it.Text == "" || (len(it.IDs) == 0 && len(it.Key) == 0) {
		return info, nil
	}
	// A learned key's source is its own landmark's cell exactly; only reads
	// go through the soft mean over landmarks.
	lm := -1
	if len(it.Key) > 0 && !h.Opt.NoLearn {
		lm = h.K.Learn(h.G, it.Key)
	}
	var v int
	if lm >= 0 {
		v = h.K.LandmarkVertex(h.G, lm)
	} else {
		var err error
		if v, err = h.Vertex(it.Key, it.IDs); err != nil {
			return info, err
		}
	}
	cls := 0
	if len(it.Key) > 0 {
		cls = h.K.Class(it.Key)
	} else if h.Sp.K > 1 {
		cls = phi(append([]int{-1}, it.IDs...), h.Sp.K)
	}
	// Salience: the share of the write's distinct tokens the memory has not
	// seen. A wholly new write enters at µ; a repeat enters one offset later.
	sal := 0
	if h.Opt.Salience > 0 && len(it.IDs) > 0 {
		distinct, fresh := map[int]bool{}, 0
		for _, id := range it.IDs {
			if distinct[id] {
				continue
			}
			distinct[id] = true
			if !h.seen[id] {
				fresh++
			}
		}
		nov := float64(fresh) / float64(len(distinct))
		sal = int(math.Round(float64(h.Opt.Salience) * (1 - nov)))
		for id := range distinct {
			h.seen[id] = true
		}
	}
	b := -h.lambda*h.nWrite + sal
	h.nWrite++
	took, err := h.F.March(v, b, h.Sp.Weights(cls), it.Text, info.Hol)
	if err != nil {
		return info, err
	}
	pid := h.F.payloadID(it.Text)
	for len(h.rotOf) <= pid {
		h.rotOf = append(h.rotOf, RotIdentity(max(h.Opt.RotN, 1)))
		h.classOf = append(h.classOf, 0)
	}
	h.rotOf[pid], h.classOf[pid] = h.rot.Clone(), cls
	info.Vertex, info.Class, info.Birth, info.Salience, info.Took = v, cls, b, sal, took
	return info, nil
}

// Read is the basin nearest the query (the owner of φ(q) first).
func (h *HES) Read(key []float32, ids []int, skip func(string) bool) (Recall, bool) {
	q, err := h.Vertex(key, ids)
	if err != nil {
		return Recall{Basin: -1}, false
	}
	return h.F.Read(q, skip)
}

// Probes is how many landmark cells a read of the learned φ visits.
const Probes = 8

// ReadAll orders up to k payloads. With a learned φ the read first probes the
// cells of the query's most similar landmarks, in that order, and takes each
// cell's owner (Probe = its rank); the continued arrival at φ(q) orders the
// rest. Without landmarks it is the continued arrival alone.
func (h *HES) ReadAll(key []float32, ids []int, skip func(string) bool, k int) []Recall {
	q, err := h.Vertex(key, ids)
	if err != nil {
		return nil
	}
	rest := h.F.ReadAll(q, skip, 0)
	for i := range rest {
		rest[i].Probe = -1
	}
	if len(key) == 0 || h.K == nil || len(h.K.LM) == 0 {
		if k > 0 && len(rest) > k {
			rest = rest[:k]
		}
		return rest
	}
	var out []Recall
	taken := map[string]bool{}
	for rank, v := range h.K.Probes(h.G, key, Probes) {
		pid := h.F.P[v]
		if pid < 0 {
			continue
		}
		p := h.F.Payload[pid]
		if taken[p] || (skip != nil && skip(p)) {
			continue
		}
		taken[p] = true
		r := h.F.recall(q, v, h.F.T[v], v == q)
		r.Probe = rank
		out = append(out, r)
	}
	for _, r := range rest {
		if !taken[r.Payload] {
			out = append(out, r)
		}
	}
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// ClassOf is the speed class payload id pid was last written with.
func (h *HES) ClassOf(pid int) int {
	if pid < 0 || pid >= len(h.classOf) {
		return 0
	}
	return h.classOf[pid]
}

// RotationOf is the SO(n) holonomy recorded with payload, and the distance
// of the current channel from it: how far the path has turned since.
func (h *HES) RotationOf(payload string) (Rot, float64, bool) {
	for pid, p := range h.F.Payload {
		if p == payload && pid < len(h.rotOf) {
			return h.rotOf[pid].Clone(), h.rot.Distance(h.rotOf[pid]), true
		}
	}
	return Rot{}, 0, false
}

// SetMetricScale applies the metric schedule s (Theorem 3) and re-anchors λ
// when the scale has drifted past M*, which keeps the scaled head start Λ in
// a band instead of flooding (contraction) or sliding onto the hash plateau
// (expansion).
func (h *HES) SetMetricScale(s float64) bool {
	if s <= 0 {
		return false
	}
	h.Sp.SetScale(s)
	lam, fired := h.anchor.Update(s, float64(h.lambda))
	if fired {
		h.lambda = max(0, int(math.Round(lam)))
	}
	return fired
}

// Persistence is the H₁ readout of the current field.
func (h *HES) Persistence(minPersist, limit int) []Class {
	return h.F.Persistence(minPersist, limit)
}

// LawReport is the retention and ownership laws evaluated for this state.
type LawReport struct {
	N, D, Writes, Payloads, Live int
	Lambda                       int
	C, Exponent                  float64
	Diam, WMin                   int
	Scaled                       float64 // Λ
	Regime                       string
	HalfLifeExact                int     // Theorem 1 on sampled sources
	HalfLifeUniversal            float64 // W from the universal law
	Capacity                     float64 // M from Theorem 1
	NewestBasinRatio             float64 // eq. (9) limit
	SandwichLo, SandwichHi       int     // Theorem 6 half-lives at κλ and λ/κ
	Anchors                      int
}

// Laws evaluates the laws for this state.
func (h *HES) Laws() LawReport {
	gr := h.Growth()
	r := LawReport{N: h.G.N, D: gr.D, Writes: h.nWrite, Payloads: len(h.F.Payload), Lambda: h.lambda,
		C: gr.C, Exponent: gr.Exponent, Diam: gr.Diam, WMin: gr.WMin, Anchors: h.anchor.Events,
		NewestBasinRatio: NewestBasinRatio(gr.D)}
	for _, n := range h.F.Owners() {
		if n > 0 {
			r.Live++
		}
	}
	lam := float64(h.lambda)
	r.Scaled = ScaledLambda(gr.C, lam, h.G.N, gr.D)
	ages := int(float64(h.G.N)*math.Ln2) + 8
	tabs := h.tables(12)
	if lam > 0 {
		S := Survival(tabs, nil, lam, ages)
		r.HalfLifeExact = HalfLife(S)
		r.Capacity = Capacity(S)
		r.HalfLifeUniversal = UniversalHalfLife(r.Scaled, gr.D) * float64(h.G.N)
		lo, hi := Sandwich(tabs, lam, h.Opt.Kappa, ages)
		r.SandwichLo, r.SandwichHi = HalfLife(lo), HalfLife(hi)
	}
	r.Regime = Regime(lam, h.G.N, max(r.HalfLifeExact, 1), gr.WMin, gr.Diam, gr.D)
	return r
}

// snapshot is what Save writes. The graph and speed fields are rebuilt from
// the options' seed, so only the field, the channels and φ are stored.
type snapshot struct {
	Opt                  Options
	NWrite, Lambda       int
	G                    Perm
	Rot                  Rot
	Seen                 []int
	T, P, Par            []int
	H                    []Perm
	Payload              []string
	SrcOf, BirthOf       []int
	RotOf                []Rot
	ClassOf              []int
	K                    *KeyMap
	Scale                float64
	Anchor               Anchor
}

// Save writes the state to path.
func (h *HES) Save(path string) error {
	s := snapshot{Opt: h.Opt, NWrite: h.nWrite, Lambda: h.lambda, G: h.g, Rot: h.rot,
		T: h.F.T, P: h.F.P, Par: h.F.Par, H: h.F.H, Payload: h.F.Payload,
		SrcOf: h.F.SrcOf, BirthOf: h.F.BirthOf, RotOf: h.rotOf, ClassOf: h.classOf,
		K: h.K, Scale: h.Sp.Scale(), Anchor: *h.anchor}
	for id := range h.seen {
		s.Seen = append(s.Seen, id)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(&s); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// LoadHES reads a state Save wrote.
func LoadHES(path string) (*HES, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var s snapshot
	if err := gob.NewDecoder(f).Decode(&s); err != nil {
		return nil, fmt.Errorf("hes: %s: %w", path, err)
	}
	o := s.Opt
	o.Lambda = s.Lambda // the design rule already ran
	h, err := NewHES(o)
	if err != nil {
		return nil, err
	}
	if len(s.T) != h.G.N {
		return nil, fmt.Errorf("hes: %s holds %d vertices, the options build %d", path, len(s.T), h.G.N)
	}
	h.Opt = s.Opt
	h.nWrite, h.lambda, h.g, h.rot = s.NWrite, s.Lambda, s.G, s.Rot
	for _, id := range s.Seen {
		h.seen[id] = true
	}
	h.F.T, h.F.P, h.F.Par, h.F.H = s.T, s.P, s.Par, s.H
	h.F.Payload, h.F.SrcOf, h.F.BirthOf = s.Payload, s.SrcOf, s.BirthOf
	h.rotOf, h.classOf, h.K = s.RotOf, s.ClassOf, s.K
	if s.Scale > 0 {
		h.Sp.SetScale(s.Scale)
	}
	a := s.Anchor
	h.anchor = &a
	return h, nil
}
