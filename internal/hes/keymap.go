package hes

import (
	"fmt"
	"math"
	"math/rand"
)

// KeyMap is φ: a key vector to a source vertex. The HES prototype projects the
// key to D coordinates with a fixed random matrix, makes each coordinate
// uniform with the normal CDF, and takes the cell (eikonal_memory.py cell_of).
// Nearby keys land in nearby cells, so a query that is not the stored key
// still reaches the stored basin. The note names φ as the bottleneck and a
// fitted φ as the next test: Fit replaces the random rows with the keys' own
// leading principal directions.
//
// The same map gives each write its token class (the speed field it marches
// with, speed.go) and the two Householder directions of the orthogonal
// holonomy step (holonomy.go). All three depend on the write's own key only,
// which is the token-local case of the ownership law (Theorem 5).
type KeyMap struct {
	M      int         // key dimension
	D      int         // torus dimension
	P      [][]float32 // D rows of length M: the projection
	Mean   []float64   // per coordinate, subtracted before the CDF
	Std    []float64   // per coordinate, divided before the CDF
	Fitted bool        // P came from Fit
	Cls    [][]float32 // class directions: argmax picks the speed field
	HA, HB [][]float32 // RotN rows each: the Householder directions
	Seed   int64

	// Learned φ. Landmarks are keys pinned to torus cells as they are learned
	// (Learn); a key maps to the softmax-weighted circular mean of its nearest
	// landmarks' cells. At most MaxLM are kept, so φ stays a fixed-size map.
	LM    [][]float32 // landmark keys, unit length
	LP    [][]float64 // their coordinates in [0,1)^D (cell centres)
	MaxLM int
	Beta  float64 // softmax sharpness over cosine similarity
	TopK  int     // landmarks in a read's mean
	Tau   float64 // a new landmark this similar to an old one sits beside it
	Dup   float64 // a key this similar to a landmark is not learned again
}

func unit32(key []float32) []float32 {
	var ss float64
	for _, v := range key {
		ss += float64(v) * float64(v)
	}
	if ss == 0 {
		return nil
	}
	inv := float32(1 / math.Sqrt(ss))
	out := make([]float32, len(key))
	for i, v := range key {
		out[i] = v * inv
	}
	return out
}

// nearest is the landmarks most similar to the unit key u, best first.
func (k *KeyMap) nearest(u []float32, n int) ([]int, []float64) {
	idx := make([]int, 0, n)
	sim := make([]float64, 0, n)
	for i, l := range k.LM {
		s := dot32(l, u)
		j := len(idx)
		for j > 0 && sim[j-1] < s {
			j--
		}
		if j >= n {
			continue
		}
		idx = append(idx, 0)
		sim = append(sim, 0)
		copy(idx[j+1:], idx[j:])
		copy(sim[j+1:], sim[j:])
		idx[j], sim[j] = i, s
		if len(idx) > n {
			idx, sim = idx[:n], sim[:n]
		}
	}
	return idx, sim
}

// circMean is the weighted circular mean of coordinates on the unit torus.
func circMean(pts [][]float64, w []float64, d int) []float64 {
	out := make([]float64, d)
	for a := 0; a < d; a++ {
		var c, s float64
		for i, p := range pts {
			th := 2 * math.Pi * p[a]
			c += w[i] * math.Cos(th)
			s += w[i] * math.Sin(th)
		}
		u := math.Atan2(s, c) / (2 * math.Pi)
		if u < 0 {
			u++
		}
		if u >= 1 {
			u = math.Nextafter(1, 0)
		}
		out[a] = u
	}
	return out
}

func (k *KeyMap) lmDefaults() {
	if k.MaxLM == 0 {
		k.MaxLM = 512
	}
	if k.Beta == 0 {
		k.Beta = 200
	}
	if k.TopK == 0 {
		k.TopK = 4
	}
	if k.Tau == 0 {
		k.Tau = -1 // every landmark after the first sits beside its likes
	}
	if k.Dup == 0 {
		// Only a practically identical key shares a landmark: two keys on one
		// cell collide, and the later write loses its source.
		k.Dup = 0.99999
	}
}

// Learn pins key as a landmark on g, unless φ is full or already holds a key
// this similar. A key like earlier landmarks is placed at their weighted
// circular mean; an unlike key at the projection's cell. Either way it takes
// the nearest cell no landmark holds. It returns the landmark that stands for
// key (the new one, or the practically identical one), or -1 when φ is full.
func (k *KeyMap) Learn(g *Graph, key []float32) int {
	k.lmDefaults()
	u := unit32(key)
	if u == nil {
		return -1
	}
	idx, sim := k.nearest(u, k.TopK)
	if len(idx) > 0 && sim[0] >= k.Dup {
		return idx[0]
	}
	if len(k.LM) >= k.MaxLM {
		return -1
	}
	var pos []float64
	if len(idx) > 0 {
		if sim[0] >= k.Tau {
			var pts [][]float64
			var w []float64
			for i, j := range idx {
				pts = append(pts, k.LP[j])
				w = append(w, math.Exp(k.Beta*(sim[i]-sim[0])))
			}
			pos = circMean(pts, w, k.D)
		}
	}
	if pos == nil {
		pos = k.project(u)
	}
	held := map[int]bool{}
	for _, p := range k.LP {
		held[cellOf(g, p)] = true
	}
	v := cellOf(g, pos)
	if held[v] {
		v = nearestFree(g, v, held)
	}
	k.LM = append(k.LM, u)
	k.LP = append(k.LP, centreOf(g, v))
	return len(k.LM) - 1
}

// Probes are the cells of the n landmarks most like key, most alike first:
// a multi-probe read of the learned φ.
func (k *KeyMap) Probes(g *Graph, key []float32, n int) []int {
	u := unit32(key)
	if u == nil || len(k.LM) == 0 {
		return nil
	}
	idx, _ := k.nearest(u, n)
	out := make([]int, len(idx))
	for i, j := range idx {
		out[i] = cellOf(g, k.LP[j])
	}
	return out
}

// LandmarkVertex is landmark i's cell on g.
func (k *KeyMap) LandmarkVertex(g *Graph, i int) int { return cellOf(g, k.LP[i]) }

func cellOf(g *Graph, u []float64) int {
	c := make([]int, len(g.Dims))
	for a, side := range g.Dims {
		if a < len(u) {
			c[a] = int(u[a] * float64(side))
		}
	}
	return g.Vertex(c)
}

func centreOf(g *Graph, v int) []float64 {
	c := make([]int, len(g.Dims))
	g.coords(v, c)
	out := make([]float64, len(g.Dims))
	for a, side := range g.Dims {
		out[a] = (float64(c[a]) + 0.5) / float64(side)
	}
	return out
}

// nearestFree is the closest vertex (by hops) not in held.
func nearestFree(g *Graph, v int, held map[int]bool) int {
	seen := map[int]bool{v: true}
	q := []int{v}
	for len(q) > 0 {
		x := q[0]
		q = q[1:]
		if !held[x] {
			return x
		}
		g.Neighbors(x, func(y, _ int) {
			if !seen[y] {
				seen[y] = true
				q = append(q, y)
			}
		})
	}
	return v
}

// project is the prototype's φ: the projection and the normal CDF.
func (k *KeyMap) project(key []float32) []float64 {
	out := make([]float64, k.D)
	for i := 0; i < k.D; i++ {
		z := (dot32(k.P[i], key) - k.Mean[i]) / k.Std[i]
		u := 0.5 * math.Erfc(-z/math.Sqrt2)
		if u >= 1 {
			u = math.Nextafter(1, 0)
		}
		if u < 0 {
			u = 0
		}
		out[i] = u
	}
	return out
}

// NewKeyMap draws the random projection. classes and rotN may be 0.
func NewKeyMap(m, d, classes, rotN int, seed int64) (*KeyMap, error) {
	if m < 1 || d < 1 {
		return nil, fmt.Errorf("hes: key map %d -> %d", m, d)
	}
	rng := rand.New(rand.NewSource(seed))
	gauss := func(rows int) [][]float32 {
		out := make([][]float32, rows)
		s := 1 / math.Sqrt(float64(m))
		for i := range out {
			out[i] = make([]float32, m)
			for j := range out[i] {
				out[i][j] = float32(rng.NormFloat64() * s)
			}
		}
		return out
	}
	k := &KeyMap{M: m, D: d, P: gauss(d), Seed: seed}
	k.Mean = make([]float64, d)
	k.Std = make([]float64, d)
	for i := range k.Std {
		// A unit key against rows of variance 1/m projects to variance 1/m.
		k.Std[i] = 1 / math.Sqrt(float64(m))
	}
	if classes > 1 {
		k.Cls = gauss(classes)
	}
	if rotN > 1 {
		k.HA = gauss(rotN)
		k.HB = gauss(rotN)
	}
	return k, nil
}

func dot32(a []float32, b []float32) float64 {
	var s float64
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// Coords are the D coordinates of key in [0, 1): through the learned
// landmarks when φ has any, else the prototype's projection.
func (k *KeyMap) Coords(key []float32) []float64 {
	if len(k.LM) > 0 {
		k.lmDefaults()
		if u := unit32(key); u != nil {
			idx, sim := k.nearest(u, k.TopK)
			var pts [][]float64
			var w []float64
			for i, j := range idx {
				pts = append(pts, k.LP[j])
				w = append(w, math.Exp(k.Beta*(sim[i]-sim[0])))
			}
			return circMean(pts, w, k.D)
		}
	}
	return k.project(key)
}

// Vertex is φ(key) on g. The torus dimension must be the map's D.
func (k *KeyMap) Vertex(g *Graph, key []float32) int {
	u := k.Coords(key)
	c := make([]int, len(g.Dims))
	for a, side := range g.Dims {
		if a < len(u) {
			c[a] = int(u[a] * float64(side))
		}
	}
	return g.Vertex(c)
}

// Class is the write's speed-field class: the class direction the key is
// most aligned with. 0 when there are no classes.
func (k *KeyMap) Class(key []float32) int {
	best, bi := math.Inf(-1), 0
	for i, c := range k.Cls {
		if s := dot32(c, key); s > best {
			best, bi = s, i
		}
	}
	return bi
}

// Householder is the orthogonal holonomy step of key: two reflections.
func (k *KeyMap) Householder(key []float32) Rot {
	n := len(k.HA)
	if n < 2 {
		return RotIdentity(max(n, 1))
	}
	a := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		a[i] = dot32(k.HA[i], key)
		b[i] = dot32(k.HB[i], key)
	}
	return Householder2(n, a, b)
}

// Calibrate sets the per-coordinate mean and spread from sample keys, so the
// CDF spreads the keys the model actually makes over the whole torus rather
// than the middle of it.
func (k *KeyMap) Calibrate(keys [][]float32) {
	if len(keys) < 2 {
		return
	}
	for i := 0; i < k.D; i++ {
		var s, ss float64
		for _, key := range keys {
			z := dot32(k.P[i], key)
			s += z
			ss += z * z
		}
		n := float64(len(keys))
		mean := s / n
		v := ss/n - mean*mean
		if v <= 1e-18 {
			continue
		}
		k.Mean[i] = mean
		k.Std[i] = math.Sqrt(v)
	}
}

// Fit makes φ the keys' own top-D principal directions (power iteration on
// the Gram matrix with deflation), then calibrates on the same keys. With
// fewer than D+1 keys the random rows stay.
func (k *KeyMap) Fit(keys [][]float32) {
	n := len(keys)
	if n < k.D+1 {
		k.Calibrate(keys)
		return
	}
	mean := make([]float64, k.M)
	for _, key := range keys {
		for j := 0; j < k.M && j < len(key); j++ {
			mean[j] += float64(key[j])
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}
	x := make([][]float64, n)
	for i, key := range keys {
		x[i] = make([]float64, k.M)
		for j := 0; j < k.M && j < len(key); j++ {
			x[i][j] = float64(key[j]) - mean[j]
		}
	}
	gram := make([][]float64, n)
	for i := range gram {
		gram[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			var s float64
			for t := 0; t < k.M; t++ {
				s += x[i][t] * x[j][t]
			}
			gram[i][j], gram[j][i] = s, s
		}
	}
	rng := rand.New(rand.NewSource(k.Seed + 1))
	var found [][]float64
	for c := 0; c < k.D; c++ {
		u := make([]float64, n)
		for i := range u {
			u[i] = rng.NormFloat64()
		}
		var lam float64
		for it := 0; it < 200; it++ {
			w := make([]float64, n)
			for i := 0; i < n; i++ {
				var s float64
				for j := 0; j < n; j++ {
					s += gram[i][j] * u[j]
				}
				w[i] = s
			}
			for _, f := range found {
				var p float64
				for i := range w {
					p += w[i] * f[i]
				}
				for i := range w {
					w[i] -= p * f[i]
				}
			}
			var ss float64
			for _, v := range w {
				ss += v * v
			}
			if ss == 0 {
				break
			}
			nrm := math.Sqrt(ss)
			var diff float64
			for i := range w {
				w[i] /= nrm
				diff += math.Abs(w[i] - u[i])
			}
			u, lam = w, nrm
			if diff < 1e-10 {
				break
			}
		}
		if lam <= 0 {
			break
		}
		found = append(found, u)
		// Direction in key space: Xᵀu, unit length.
		dir := make([]float64, k.M)
		for i := 0; i < n; i++ {
			for t := 0; t < k.M; t++ {
				dir[t] += x[i][t] * u[i]
			}
		}
		var ss float64
		for _, v := range dir {
			ss += v * v
		}
		if ss == 0 {
			break
		}
		inv := 1 / math.Sqrt(ss)
		row := make([]float32, k.M)
		for t := range dir {
			row[t] = float32(dir[t] * inv)
		}
		k.P[c] = row
	}
	k.Fitted = len(found) == k.D
	k.Calibrate(keys)
}
