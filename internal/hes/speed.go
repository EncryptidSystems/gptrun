package hes

import (
	"math"
	"math/rand"
)

// Speeds is the speed field F_t = f(x_t) of the HES write rule: the token bends
// the metric. It is the token-local form of the ownership paper (§6.5): K
// token classes, each with a smooth random field f_k on the vertices (white
// noise smoothed by a Gaussian of width Sigma cells, standardized), and under
// class k an edge (u, v) weighs
//
//	round(w_uv · κ^tanh((f_k(u) + f_k(v)) / 2))
//
// so every multiplier lies in (1/κ, κ). A write's metric depends on its own
// token only, which is the case Theorem 5 of that paper makes exact; any
// metric within κ of the base obeys the Theorem 6 sandwich.
type Speeds struct {
	K     int
	Kappa float64
	Sigma float64
	F     [][]float64 // K fields over the vertices
	W     [][]int     // K weight tables over the edges, at metric scale 1
	scale float64
	cur   [][]int // W at the current metric scale
}

// NewSpeeds draws K fields on g. K < 1 or κ ≤ 1 is the base metric alone.
func NewSpeeds(g *Graph, k int, kappa, sigma float64, seed int64) *Speeds {
	s := &Speeds{K: k, Kappa: kappa, Sigma: sigma, scale: 1}
	if k < 1 || kappa <= 1 {
		s.K = 1
		s.W = [][]int{append([]int(nil), g.W...)}
		s.F = [][]float64{make([]float64, g.N)}
		s.cur = s.W
		return s
	}
	rng := rand.New(rand.NewSource(seed))
	for c := 0; c < k; c++ {
		f := make([]float64, g.N)
		for i := range f {
			f[i] = rng.NormFloat64()
		}
		f = smoothTorus(g, f, sigma)
		standardize(f)
		w := make([]int, len(g.Edges))
		for e, uv := range g.Edges {
			m := math.Pow(kappa, math.Tanh((f[uv[0]]+f[uv[1]])/2))
			w[e] = max(1, int(math.Round(float64(g.W[e])*m)))
		}
		s.F = append(s.F, f)
		s.W = append(s.W, w)
	}
	s.cur = s.W
	return s
}

// Weights is the edge table of class c at the current metric scale.
func (s *Speeds) Weights(c int) []int {
	if c < 0 || c >= len(s.cur) {
		c = 0
	}
	return s.cur[c]
}

// MaxWeight is the heaviest edge over all classes at the current scale: the
// key window C a march needs.
func (s *Speeds) MaxWeight() int {
	m := 1
	for _, w := range s.cur {
		for _, x := range w {
			if x > m {
				m = x
			}
		}
	}
	return m
}

// SetScale is the metric schedule of the retention paper's Theorem 3: every
// edge is multiplied by sc. sc must be positive.
func (s *Speeds) SetScale(sc float64) {
	if sc <= 0 || sc == s.scale {
		return
	}
	s.scale = sc
	if sc == 1 {
		s.cur = s.W
		return
	}
	s.cur = make([][]int, len(s.W))
	for c, w := range s.W {
		t := make([]int, len(w))
		for e, x := range w {
			t[e] = max(1, int(math.Round(float64(x)*sc)))
		}
		s.cur[c] = t
	}
}

// Scale is the current metric scale.
func (s *Speeds) Scale() float64 { return s.scale }

// smoothTorus convolves f with a Gaussian of width sigma cells along each axis
// of the torus (separable, wrapping).
func smoothTorus(g *Graph, f []float64, sigma float64) []float64 {
	if sigma <= 0 {
		return f
	}
	r := int(math.Ceil(3 * sigma))
	ker := make([]float64, 2*r+1)
	var z float64
	for i := -r; i <= r; i++ {
		ker[i+r] = math.Exp(-float64(i*i) / (2 * sigma * sigma))
		z += ker[i+r]
	}
	for i := range ker {
		ker[i] /= z
	}
	cur := f
	coord := make([]int, len(g.Dims))
	for a, side := range g.Dims {
		if side < 2 {
			continue
		}
		next := make([]float64, g.N)
		for v := 0; v < g.N; v++ {
			g.coords(v, coord)
			var s float64
			c0 := coord[a]
			for i := -r; i <= r; i++ {
				coord[a] = c0 + i
				s += ker[i+r] * cur[g.Vertex(coord)]
			}
			coord[a] = c0
			next[v] = s
		}
		cur = next
	}
	return cur
}

func standardize(f []float64) {
	var s, ss float64
	for _, v := range f {
		s += v
		ss += v * v
	}
	n := float64(len(f))
	mean := s / n
	sd := math.Sqrt(math.Max(ss/n-mean*mean, 1e-18))
	for i := range f {
		f[i] = (f[i] - mean) / sd
	}
}
