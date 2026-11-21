package hes

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// Graph is the latent complex K of the HES design note: N vertices, never
// grown, with positive integer edge weights. A torus also carries its square
// 2-cells, so the sublevel sets {T ≤ τ} have an H₁ that faces can fill
// (persist.go).
//
// The retention and ownership laws are stated on D-dimensional tori with
// integer weights drawn from ±20% of a base weight; Torus builds exactly that
// graph, seeded, so a saved state rebuilds the same metric.
type Graph struct {
	N     int
	Dims  []int     // torus side lengths; their product is N
	Edges [][2]int  // undirected, u < v not required
	W     []int     // base weight of each edge
	Axis  []int     // the torus axis an edge runs along
	Faces [][4]int  // square 2-cells as four edge ids, in boundary order
	off   []int     // CSR: neighbours of v are nbr[off[v]:off[v+1]]
	nbr   []int     // neighbour vertex
	eid   []int     // edge id of that neighbour link
	Base  int       // base weight
	Seed  int64     // the seed the weights came from
	Jit   float64   // the jitter (0.2 = ±20%)
	wmin  int
	wmax  int
}

// Torus builds the D-dimensional torus with the given side lengths. Each edge
// weight is drawn uniformly from [w(1-jit), w(1+jit)] with the seed. A side of
// length 1 adds no edges along that axis; a side of 2 adds one edge, not a
// double edge.
func Torus(dims []int, w int, jit float64, seed int64) (*Graph, error) {
	if len(dims) == 0 {
		return nil, fmt.Errorf("hes: torus needs at least one side")
	}
	n := 1
	for _, d := range dims {
		if d < 1 {
			return nil, fmt.Errorf("hes: torus side %d", d)
		}
		n *= d
	}
	if w < 1 {
		return nil, fmt.Errorf("hes: base weight %d", w)
	}
	if jit < 0 || jit >= 1 {
		return nil, fmt.Errorf("hes: jitter %g", jit)
	}
	g := &Graph{N: n, Dims: append([]int(nil), dims...), Base: w, Seed: seed, Jit: jit}
	rng := rand.New(rand.NewSource(seed))
	lo := int(math.Round(float64(w) * (1 - jit)))
	hi := int(math.Round(float64(w) * (1 + jit)))
	if lo < 1 {
		lo = 1
	}
	if hi < lo {
		hi = lo
	}
	edgeAt := map[[2]int]int{} // (vertex, axis) -> edge from vertex one step up that axis
	coord := make([]int, len(dims))
	for v := 0; v < n; v++ {
		g.coords(v, coord)
		for a, side := range dims {
			if side < 2 || (side == 2 && coord[a] == 1) {
				continue
			}
			u := g.step(v, a, 1)
			wt := lo + rng.Intn(hi-lo+1)
			edgeAt[[2]int{v, a}] = len(g.Edges)
			g.Edges = append(g.Edges, [2]int{v, u})
			g.W = append(g.W, wt)
			g.Axis = append(g.Axis, a)
		}
	}
	// Square faces: for each vertex and pair of axes a < b, the cell
	// v → v+a → v+a+b → v+b → v.
	for v := 0; v < n; v++ {
		for a := 0; a < len(dims); a++ {
			for b := a + 1; b < len(dims); b++ {
				if dims[a] < 3 || dims[b] < 3 {
					continue
				}
				va := g.step(v, a, 1)
				vb := g.step(v, b, 1)
				e1, ok1 := edgeAt[[2]int{v, a}]
				e2, ok2 := edgeAt[[2]int{va, b}]
				e3, ok3 := edgeAt[[2]int{vb, a}]
				e4, ok4 := edgeAt[[2]int{v, b}]
				if ok1 && ok2 && ok3 && ok4 {
					g.Faces = append(g.Faces, [4]int{e1, e2, e3, e4})
				}
			}
		}
	}
	g.index()
	return g, nil
}

func (g *Graph) index() {
	deg := make([]int, g.N+1)
	for _, e := range g.Edges {
		deg[e[0]]++
		deg[e[1]]++
	}
	g.off = make([]int, g.N+1)
	for v := 0; v < g.N; v++ {
		g.off[v+1] = g.off[v] + deg[v]
	}
	g.nbr = make([]int, g.off[g.N])
	g.eid = make([]int, g.off[g.N])
	fill := append([]int(nil), g.off[:g.N]...)
	for i, e := range g.Edges {
		u, v := e[0], e[1]
		g.nbr[fill[u]], g.eid[fill[u]] = v, i
		fill[u]++
		g.nbr[fill[v]], g.eid[fill[v]] = u, i
		fill[v]++
	}
	g.wmin, g.wmax = 0, 0
	for i, w := range g.W {
		if i == 0 || w < g.wmin {
			g.wmin = w
		}
		if w > g.wmax {
			g.wmax = w
		}
	}
}

// CSR is the adjacency: the neighbours of v are nbr[off[v]:off[v+1]], each
// through edge eid[i]. The slices are the graph's own; do not change them.
func (g *Graph) CSR() (off, nbr, eid []int) { return g.off, g.nbr, g.eid }

// Dim is the torus dimension D.
func (g *Graph) Dim() int { return len(g.Dims) }

// WMin and WMax are the lightest and heaviest base edges.
func (g *Graph) WMin() int { return g.wmin }
func (g *Graph) WMax() int { return g.wmax }

func (g *Graph) coords(v int, out []int) {
	for a := len(g.Dims) - 1; a >= 0; a-- {
		out[a] = v % g.Dims[a]
		v /= g.Dims[a]
	}
}

// Vertex is the vertex at the given coordinates (each taken modulo its side).
func (g *Graph) Vertex(coord []int) int {
	v := 0
	for a, side := range g.Dims {
		c := coord[a] % side
		if c < 0 {
			c += side
		}
		v = v*side + c
	}
	return v
}

func (g *Graph) step(v, axis, d int) int {
	c := make([]int, len(g.Dims))
	g.coords(v, c)
	c[axis] += d
	return g.Vertex(c)
}

// Neighbors calls f for each neighbour of v with the edge id.
func (g *Graph) Neighbors(v int, f func(u, e int)) {
	for i := g.off[v]; i < g.off[v+1]; i++ {
		f(g.nbr[i], g.eid[i])
	}
}

// Other is the end of edge e that is not v.
func (g *Graph) Other(e, v int) int {
	if g.Edges[e][0] == v {
		return g.Edges[e][1]
	}
	return g.Edges[e][0]
}

// Distances is the shortest-path row from s under weights w (nil = base).
func (g *Graph) Distances(s int, w []int) []int {
	if w == nil {
		w = g.W
	}
	d := make([]int, g.N)
	for i := range d {
		d[i] = noTime
	}
	maxw := 1
	for _, x := range w {
		if x > maxw {
			maxw = x
		}
	}
	h, err := newMRH(RadicesSmallToLarge(maxw), maxw)
	if err != nil {
		panic(err)
	}
	d[s] = 0
	inHeap := make([]bool, g.N)
	if err := h.insert(s, 0); err != nil {
		panic(err)
	}
	inHeap[s] = true
	for h.n > 0 {
		u, k, _ := h.extract()
		inHeap[u] = false
		for i := g.off[u]; i < g.off[u+1]; i++ {
			v := g.nbr[i]
			nd := k + w[g.eid[i]]
			if nd >= d[v] {
				continue
			}
			d[v] = nd
			if inHeap[v] {
				if err := h.decrease(v, nd); err != nil {
					panic(err)
				}
				continue
			}
			if err := h.insert(v, nd); err != nil {
				panic(err)
			}
			inHeap[v] = true
		}
	}
	return d
}

// SortedRow is Distances sorted ascending: |B_s(r)| is the number of entries
// below r (sort.SearchInts).
func (g *Graph) SortedRow(s int, w []int) []int {
	d := g.Distances(s, w)
	sort.Ints(d)
	return d
}

// RadicesSmallToLarge is the radix tuple the HES note found cheapest at a
// fixed product: small radices at the fine levels and one large radix at the
// top. The fine levels are 2, 4, 8; the top radix makes the product cover C.
func RadicesSmallToLarge(c int) []int {
	if c < 1 {
		c = 1
	}
	r := []int{}
	p := 1
	for _, x := range []int{2, 4, 8} {
		if p >= c+1 {
			break
		}
		r = append(r, x)
		p *= x
	}
	if p < c+1 {
		top := (c + p) / p // ceil((c+1)/p)
		if top < 2 {
			top = 2
		}
		r = append(r, top)
	}
	return r
}
