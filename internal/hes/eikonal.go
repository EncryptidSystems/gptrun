package hes

import (
	"fmt"
	"sort"
)

// Eik is channel 2 of the HES note on a latent complex K (graph.go). Each
// vertex v holds
//
//	T(v)   the arrival time, the min over writes of birth + geodesic cost,
//	p(v)   the payload of the write whose front reached v first,
//	H(v)   the holonomy along that geodesic, H(v) = H(u)·P_uv,
//	par(v) the vertex the front came from: the characteristic, so a read
//	       follows −∇T back to the basin without searching.
//
// A write marches from its source with its own metric (the speed field of
// its token class) on the multi-radix heap, keyed by cost from the source,
// with window C = the heaviest edge of that metric, and relaxes only strict
// improvements. That is the pruned incremental writer of the ownership paper:
// with one fixed metric it computes exactly T = min_i(b_i + d(s_i, v)), the
// Johnson–Mehl tessellation, and with token metrics it keeps every item the
// analytic field keeps (its Proposition 6).
type Eik struct {
	G       *Graph
	Rank    int
	T       []int
	P       []int // payload id, -1 where no front arrived
	Par     []int // characteristic parent, -1 at a source
	H       []Perm
	Payload []string
	SrcOf   []int // per payload id: the source vertex of its latest write
	BirthOf []int // per payload id: that write's birth key
	Radices []int // nil: RadicesSmallToLarge of each write's window
	// march scratch, reused across writes
	tent    []int
	stamp   []int
	parent  []int
	settled []int
	epoch   int
}

// NewEik is an empty field on g.
func NewEik(g *Graph, rank int) (*Eik, error) {
	if g == nil || g.N < 1 {
		return nil, fmt.Errorf("hes: empty complex")
	}
	if rank < 2 {
		return nil, fmt.Errorf("hes: rank %d", rank)
	}
	f := &Eik{G: g, Rank: rank,
		T: make([]int, g.N), P: make([]int, g.N), Par: make([]int, g.N), H: make([]Perm, g.N),
		tent: make([]int, g.N), stamp: make([]int, g.N), parent: make([]int, g.N), settled: make([]int, g.N)}
	for v := range f.T {
		f.T[v], f.P[v], f.Par[v] = noTime, -1, -1
		f.H[v] = Identity(rank)
	}
	return f, nil
}

// Transport is P_uv for edge e traversed from u: the transposition of the two
// ends' letters, as the cycle field used. It is its own inverse, so P_vu =
// P_uv⁻¹, and a square cell's four transports need not multiply to e: the
// connection has curvature, which is what a 1-cycle's holonomy detects.
func (f *Eik) Transport(u, v int) Perm {
	a, b := u%f.Rank, v%f.Rank
	if a == b {
		return Identity(f.Rank)
	}
	return Transposition(f.Rank, a, b)
}

func (f *Eik) payloadID(s string) int {
	for i, p := range f.Payload {
		if p == s {
			return i
		}
	}
	f.Payload = append(f.Payload, s)
	f.SrcOf = append(f.SrcOf, -1)
	f.BirthOf = append(f.BirthOf, 0)
	return len(f.Payload) - 1
}

// March writes one source: vertex src, birth key b (−λ·i + salience), edge
// weights w (the token's metric), payload and the holonomy at the source.
// It returns how many vertices the front took.
func (f *Eik) March(src, b int, w []int, payload string, hol Perm) (int, error) {
	g := f.G
	if src < 0 || src >= g.N {
		return 0, fmt.Errorf("hes: vertex %d outside 0..%d", src, g.N-1)
	}
	if len(w) != len(g.Edges) {
		return 0, fmt.Errorf("hes: %d weights for %d edges", len(w), len(g.Edges))
	}
	if len(hol) == 0 {
		hol = Identity(f.Rank)
	}
	if len(hol) != f.Rank {
		return 0, fmt.Errorf("hes: holonomy rank %d, field rank %d", len(hol), f.Rank)
	}
	pid := f.payloadID(payload)
	f.SrcOf[pid], f.BirthOf[pid] = src, b
	if b >= f.T[src] {
		return 0, nil
	}
	c := 1
	for _, x := range w {
		if x > c {
			c = x
		}
	}
	r := f.Radices
	if r == nil {
		r = RadicesSmallToLarge(c)
	}
	h, err := newMRH(r, c)
	if err != nil {
		return 0, err
	}
	f.epoch++
	ep := f.epoch
	f.stamp[src], f.tent[src], f.parent[src] = ep, 0, -1
	inHeap := map[int]bool{src: true}
	if err := h.insert(src, 0); err != nil {
		return 0, err
	}
	took := 0
	for h.n > 0 {
		x, key, ok := h.extract()
		if !ok {
			break
		}
		delete(inHeap, x)
		if f.settled[x] == ep {
			continue
		}
		f.settled[x] = ep
		abs := b + key
		if abs >= f.T[x] {
			continue
		}
		f.T[x], f.P[x] = abs, pid
		if p := f.parent[x]; p < 0 {
			f.Par[x] = -1
			f.H[x] = append(Perm(nil), hol...)
		} else {
			f.Par[x] = p
			f.H[x] = f.H[p].Mul(f.Transport(p, x))
		}
		took++
		for i := g.off[x]; i < g.off[x+1]; i++ {
			v := g.nbr[i]
			if f.settled[v] == ep {
				continue
			}
			nd := key + w[g.eid[i]]
			if b+nd >= f.T[v] || (f.stamp[v] == ep && nd >= f.tent[v]) {
				continue
			}
			f.stamp[v], f.tent[v], f.parent[v] = ep, nd, x
			if inHeap[v] {
				if err := h.decrease(v, nd); err != nil {
					return took, err
				}
				continue
			}
			if err := h.insert(v, nd); err != nil {
				return took, err
			}
			inHeap[v] = true
		}
	}
	return took, nil
}

// Basin follows the characteristic from v back to its source (−∇T), inside
// v's own payload: a later write may have taken a parent, and the walk does
// not cross into that territory. When the chain is cut that way, the basin is
// the payload's source if it still owns it, else the last vertex reached.
func (f *Eik) Basin(v int) int {
	if v < 0 || v >= f.G.N || f.P[v] < 0 {
		return v
	}
	pid := f.P[v]
	for i := 0; i < f.G.N; i++ {
		p := f.Par[v]
		if p < 0 {
			return v
		}
		if f.P[p] != pid {
			break
		}
		v = p
	}
	if s := f.SrcOf[pid]; s >= 0 && f.P[s] == pid {
		return s
	}
	return v
}

// Recall is one read.
type Recall struct {
	Payload string
	Query   int  // the query vertex φ(q)
	Vertex  int  // where the read entered the payload's territory
	Basin   int  // the source the characteristic leads back to
	Arrival int  // T(Vertex) + d(q, Vertex): the continued arrival at q
	Owner   bool // the payload owns the query vertex itself
	Probe   int  // the landmark rank that found it (HES.ReadAll), −1 for the continued arrival
	Hol     Perm // H at the basin
	HolAt   Perm // H at the entry vertex
}

// Read is the payload at the query vertex, by geodesic nearest source: the
// owner of q when skip allows it, otherwise the allowed payload whose
// territory the arrival field continued to q reaches first.
func (f *Eik) Read(q int, skip func(string) bool) (Recall, bool) {
	all := f.ReadAll(q, skip, 1)
	if len(all) == 0 {
		return Recall{Query: q, Basin: -1}, false
	}
	return all[0], true
}

// ReadAll orders up to k payloads (k ≤ 0: all) by their continued arrival at
// q. The owner of q comes first. Every other payload p scores
// min over its vertices u of T(u) + d(q, u), the arrival field continued from
// its territory to q under the base metric; ties go to the lower vertex.
func (f *Eik) ReadAll(q int, skip func(string) bool, k int) []Recall {
	g := f.G
	if q < 0 || q >= g.N {
		return nil
	}
	allowed := func(pid int) bool {
		return pid >= 0 && (skip == nil || !skip(f.Payload[pid]))
	}
	var out []Recall
	first := -1
	if allowed(f.P[q]) {
		first = f.P[q]
		out = append(out, f.recall(q, q, f.T[q], true))
		if k == 1 {
			return out
		}
	}
	d := g.Distances(q, nil)
	best := map[int]int{}
	at := map[int]int{}
	for u := 0; u < g.N; u++ {
		pid := f.P[u]
		if pid == first || !allowed(pid) || d[u] == noTime || f.T[u] == noTime {
			continue
		}
		s := f.T[u] + d[u]
		if b, ok := best[pid]; !ok || s < b || (s == b && u < at[pid]) {
			best[pid], at[pid] = s, u
		}
	}
	pids := make([]int, 0, len(best))
	for p := range best {
		pids = append(pids, p)
	}
	sort.Slice(pids, func(i, j int) bool {
		a, b := pids[i], pids[j]
		if best[a] != best[b] {
			return best[a] < best[b]
		}
		return at[a] < at[b]
	})
	for _, p := range pids {
		if k > 0 && len(out) >= k {
			break
		}
		out = append(out, f.recall(q, at[p], best[p], false))
	}
	return out
}

func (f *Eik) recall(q, u, arr int, owner bool) Recall {
	b := f.Basin(u)
	return Recall{
		Payload: f.Payload[f.P[u]], Query: q, Vertex: u, Basin: b, Arrival: arr, Owner: owner, Probe: -1,
		Hol: append(Perm(nil), f.H[b]...), HolAt: append(Perm(nil), f.H[u]...),
	}
}

// Owners counts vertices per payload. Zero means evicted.
func (f *Eik) Owners() map[string]int {
	n := map[string]int{}
	for _, p := range f.Payload {
		n[p] = 0
	}
	for _, pid := range f.P {
		if pid >= 0 {
			n[f.Payload[pid]]++
		}
	}
	return n
}

// OwnsSource reports whether payload still owns its own source vertex: the
// survival event of the retention law.
func (f *Eik) OwnsSource(payload string) bool {
	for pid, p := range f.Payload {
		if p == payload {
			s := f.SrcOf[pid]
			return s >= 0 && f.P[s] == pid
		}
	}
	return false
}

// AnalyticField is T(v) = min_i(b_i + d_i(s_i, v)) computed directly, one
// shortest-path row per write, ties to the earliest write. ws[i] nil is the
// base metric. It is the reference the incremental writer is checked against.
func AnalyticField(g *Graph, src, birth []int, ws [][]int) (T, own []int) {
	T = make([]int, g.N)
	own = make([]int, g.N)
	for v := range T {
		T[v], own[v] = noTime, -1
	}
	for i, s := range src {
		var w []int
		if i < len(ws) {
			w = ws[i]
		}
		d := g.Distances(s, w)
		for v := range T {
			if d[v] == noTime {
				continue
			}
			if a := birth[i] + d[v]; a < T[v] {
				T[v], own[v] = a, i
			}
		}
	}
	return T, own
}
