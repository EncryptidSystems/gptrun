package hes

import "sort"

// The HES note's readout: persistent H₁ of the sublevel sets {T ≤ τ} as τ
// moves through the filtration. A 1-cycle is born when two fronts close
// around a hole in K and dies when a 2-cell fills it. Combined with H(v), a
// cycle whose holonomy is not the identity is a stored obstruction: a record
// that the input path went around something, which a small change of T
// cannot erase.
//
// The complex is the torus with its square cells. A vertex enters at T(v), an
// edge at the later of its ends, a cell at the latest of its corners. Vertices
// no front reached are left out with everything on them. Reduction is the
// standard Z/2 column reduction of the cell boundaries against the edge order;
// each pivot is a positive edge (the youngest edge of a cycle closes it).

// Class is one H₁ class.
type Class struct {
	Birth, Death int  // filtration values; Death is noTime for an essential class
	Essential    bool // never filled (the torus's own loops, or a hole no cell closes)
	Edge         int  // the edge whose arrival created the cycle
	Face         int  // the cell that filled it, -1 if essential
	Cycle        []int
	// Hol is the transport around the cycle, Π P_{v_i v_{i+1}}: the curvature
	// the connection carries around this loop.
	Hol Perm
	// Defect is Π H(v_i)·P_{v_i v_{i+1}}·H(v_{i+1})⁻¹: identity along one
	// front's geodesic tree, not across a front boundary. A non-identity
	// defect is the stored obstruction.
	Defect      Perm
	Obstruction bool
	Owners      []string // the payloads whose territory the cycle passes through
}

// Persistence returns the H₁ classes whose persistence (death − birth) is at
// least minPersist, essential classes first, then by persistence, at most
// limit of them (limit ≤ 0: all).
func (f *Eik) Persistence(minPersist, limit int) []Class {
	g := f.G
	in := func(v int) bool { return f.T[v] != noTime }
	ef := make([]int, len(g.Edges))
	var edges []int
	for e, uv := range g.Edges {
		if in(uv[0]) && in(uv[1]) {
			ef[e] = max(f.T[uv[0]], f.T[uv[1]])
			edges = append(edges, e)
		}
	}
	sort.Slice(edges, func(i, j int) bool {
		a, b := edges[i], edges[j]
		if ef[a] != ef[b] {
			return ef[a] < ef[b]
		}
		return a < b
	})
	ord := make([]int, len(g.Edges))
	for i := range ord {
		ord[i] = -1
	}
	for i, e := range edges {
		ord[e] = i
	}
	// H₀ by union-find: an edge joining two components is negative; one
	// inside a component is positive and creates a 1-cycle.
	parent := make([]int, g.N)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	positive := make([]bool, len(g.Edges))
	tree := make([][]int, g.N) // forest edges at each vertex
	for _, e := range edges {
		u, v := g.Edges[e][0], g.Edges[e][1]
		a, b := find(u), find(v)
		if a == b {
			positive[e] = true
			continue
		}
		parent[a] = b
		tree[u] = append(tree[u], e)
		tree[v] = append(tree[v], e)
	}
	// Cells in filtration order, each boundary reduced.
	ff := make([]int, len(g.Faces))
	var faces []int
	for i, fc := range g.Faces {
		ok := true
		m := 0
		for _, e := range fc {
			if ord[e] < 0 {
				ok = false
				break
			}
			m = max(m, ef[e])
		}
		if ok {
			ff[i] = m
			faces = append(faces, i)
		}
	}
	sort.Slice(faces, func(i, j int) bool {
		a, b := faces[i], faces[j]
		if ff[a] != ff[b] {
			return ff[a] < ff[b]
		}
		return a < b
	})
	low := map[int][]int{} // pivot edge order -> reduced column
	killer := map[int]int{}
	for _, fi := range faces {
		col := make([]int, 0, 4)
		for _, e := range g.Faces[fi] {
			col = append(col, ord[e])
		}
		sort.Ints(col)
		col = dedupZ2(col)
		for len(col) > 0 {
			p := col[len(col)-1]
			other, ok := low[p]
			if !ok {
				break
			}
			col = symDiff(col, other)
		}
		if len(col) > 0 {
			p := col[len(col)-1]
			low[p] = col
			killer[edges[p]] = fi
		}
	}
	var out []Class
	for _, e := range edges {
		if !positive[e] {
			continue
		}
		c := Class{Birth: ef[e], Edge: e, Face: -1, Death: noTime}
		if fi, ok := killer[e]; ok {
			c.Face, c.Death = fi, ff[fi]
		} else {
			c.Essential = true
		}
		if !c.Essential && c.Death-c.Birth < minPersist {
			continue
		}
		out = append(out, c)
	}
	sort.SliceStable(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Essential != b.Essential {
			return a.Essential
		}
		pa, pb := a.Death-a.Birth, b.Death-b.Birth
		if a.Essential {
			pa, pb = -a.Birth, -b.Birth
		}
		return pa > pb
	})
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	for i := range out {
		e := out[i].Edge
		cyc := forestPath(g, tree, ord, g.Edges[e][1], g.Edges[e][0], ord[e])
		if cyc == nil {
			continue
		}
		// The walk v → … → u, then the creating edge u → v closes it.
		out[i].Cycle = cyc
		out[i].Hol, out[i].Defect = f.loopHolonomy(cyc)
		out[i].Obstruction = !out[i].Defect.IsIdentity()
		seen := map[int]bool{}
		for _, v := range cyc {
			if p := f.P[v]; p >= 0 && !seen[p] {
				seen[p] = true
				out[i].Owners = append(out[i].Owners, f.Payload[p])
			}
		}
	}
	return out
}

// loopHolonomy is the transport and the field defect around the closed walk
// cyc[0] → cyc[1] → … → cyc[0].
func (f *Eik) loopHolonomy(cyc []int) (hol, defect Perm) {
	hol, defect = Identity(f.Rank), Identity(f.Rank)
	for i := range cyc {
		a, b := cyc[i], cyc[(i+1)%len(cyc)]
		p := f.Transport(a, b)
		hol = hol.Mul(p)
		defect = defect.Mul(f.H[a].Mul(p).Mul(f.H[b].Inv()))
	}
	return hol, defect
}

// forestPath is the vertex walk from a to b in the spanning forest restricted
// to edges older than before, or nil.
func forestPath(g *Graph, tree [][]int, ord []int, a, b, before int) []int {
	prev := map[int]int{a: -1}
	queue := []int{a}
	for len(queue) > 0 {
		x := queue[0]
		queue = queue[1:]
		if x == b {
			break
		}
		for _, e := range tree[x] {
			if ord[e] >= before {
				continue
			}
			y := g.Other(e, x)
			if _, ok := prev[y]; ok {
				continue
			}
			prev[y] = x
			queue = append(queue, y)
		}
	}
	if _, ok := prev[b]; !ok {
		return nil
	}
	var path []int
	for x := b; x != -1; x = prev[x] {
		path = append(path, x)
	}
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}
	return path
}

func dedupZ2(s []int) []int {
	out := s[:0]
	for i := 0; i < len(s); {
		j := i
		for j < len(s) && s[j] == s[i] {
			j++
		}
		if (j-i)%2 == 1 {
			out = append(out, s[i])
		}
		i = j
	}
	return out
}

func symDiff(a, b []int) []int {
	out := make([]int, 0, len(a)+len(b))
	i, j := 0, 0
	for i < len(a) && j < len(b) {
		switch {
		case a[i] < b[j]:
			out = append(out, a[i])
			i++
		case a[i] > b[j]:
			out = append(out, b[j])
			j++
		default:
			i++
			j++
		}
	}
	out = append(out, a[i:]...)
	return append(out, b[j:]...)
}
