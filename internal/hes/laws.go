package hes

import (
	"math"
	"math/rand"
	"sort"
)

// The retention law (A Retention Scaling Law for Recency-Weighted Eikonal
// Memory) and the ownership law (Basins, Noise and Moving Metrics) of the
// recency-weighted field. Write i is born at b_i = −λ·i; an item is
// retrievable while it owns its source vertex (retention) or, for a noisy
// query, the vertex the query reads (ownership).

// Ball is one vertex's distance table: d sorted ascending, cum[k] the source
// mass of the k nearest vertices. Uniform mass is k/N.
type Ball struct {
	d   []int
	cum []float64
}

// NewBall is the table of v under weights w (nil = base) and source law p
// (nil = uniform).
func NewBall(g *Graph, v int, w []int, p []float64) Ball {
	dist := g.Distances(v, w)
	idx := make([]int, g.N)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return dist[idx[a]] < dist[idx[b]] })
	b := Ball{d: make([]int, g.N), cum: make([]float64, g.N+1)}
	for k, u := range idx {
		b.d[k] = dist[u]
		m := 1 / float64(g.N)
		if p != nil {
			m = p[u]
		}
		b.cum[k+1] = b.cum[k] + m
	}
	return b
}

// Open is p(B_v(r)) = mass of {u : d(v, u) < r}.
func (b Ball) Open(r float64) float64 {
	if r <= 0 {
		return 0
	}
	k := sort.Search(len(b.d), func(i int) bool { return float64(b.d[i]) >= r })
	return b.cum[k]
}

// Closed is p(B̄_v(r)) = mass of {u : d(v, u) ≤ r}, empty for r < 0.
func (b Ball) Closed(r float64) float64 {
	if r < 0 {
		return 0
	}
	k := sort.Search(len(b.d), func(i int) bool { return float64(b.d[i]) > r })
	return b.cum[k]
}

// Count is |B_v(r)|, the open ball's vertex count.
func (b Ball) Count(r float64) int {
	return sort.Search(len(b.d), func(i int) bool { return float64(b.d[i]) >= r })
}

// Growth is the ball-growth fit |B(r)| ≈ 1 + c·r^D of the retention paper,
// over 4·w ≤ r ≤ diam/4, from sampled sources.
type Growth struct {
	C        float64 // fitted with D fixed to the torus dimension
	Exponent float64 // free log-log slope over the same window
	D        int
	Diam     int // largest distance seen from the samples
	WMin     int
	N        int
}

// MeasureGrowth fits c (and the free exponent) from samples sources.
func MeasureGrowth(g *Graph, w []int, samples int, seed int64) Growth {
	if w == nil {
		w = g.W
	}
	rng := rand.New(rand.NewSource(seed))
	samples = max(1, min(samples, g.N))
	var balls []Ball
	diam := 0
	for i := 0; i < samples; i++ {
		b := NewBall(g, rng.Intn(g.N), w, nil)
		balls = append(balls, b)
		if last := b.d[len(b.d)-1]; last != noTime && last > diam {
			diam = last
		}
	}
	wmin := 0
	for i, x := range w {
		if i == 0 || x < wmin {
			wmin = x
		}
	}
	gr := Growth{D: g.Dim(), Diam: diam, WMin: wmin, N: g.N}
	lo, hi := 4*float64(g.Base), float64(diam)/4
	if hi <= lo {
		lo, hi = float64(wmin), math.Max(float64(diam)/2, float64(wmin)+1)
	}
	var xs, ys []float64
	for k := 0; k < 24; k++ {
		r := lo * math.Pow(hi/lo, float64(k)/23)
		var m float64
		for _, b := range balls {
			m += float64(b.Count(r))
		}
		m = m/float64(len(balls)) - 1
		if m > 0 {
			xs = append(xs, math.Log(r))
			ys = append(ys, math.Log(m))
		}
	}
	if len(xs) == 0 {
		return gr
	}
	var s float64
	for i := range xs {
		s += ys[i] - float64(gr.D)*xs[i]
	}
	gr.C = math.Exp(s / float64(len(xs)))
	if len(xs) > 1 {
		var mx, my float64
		for i := range xs {
			mx += xs[i]
			my += ys[i]
		}
		mx /= float64(len(xs))
		my /= float64(len(xs))
		var sxy, sxx float64
		for i := range xs {
			sxy += (xs[i] - mx) * (ys[i] - my)
			sxx += (xs[i] - mx) * (xs[i] - mx)
		}
		if sxx > 0 {
			gr.Exponent = sxy / sxx
		}
	}
	return gr
}

// Survival is Theorem 1 (uniform p) and Theorem 2 (any p): S(a) for
// a = 0..ages, averaged over the sources whose tables are given, each source
// weighted by weight[i] (nil: equal weights, the uniform average).
//
//	S(a) = Σ_s p(s) Π_{m=1..a} (1 − p(B_s(λm)))
func Survival(tables []Ball, weight []float64, lambda float64, ages int) []float64 {
	S := make([]float64, ages+1)
	var tot float64
	for i := range tables {
		w := 1.0
		if weight != nil {
			w = weight[i]
		}
		tot += w
		prod := 1.0
		S[0] += w
		for a := 1; a <= ages; a++ {
			prod *= 1 - tables[i].Open(lambda*float64(a))
			S[a] += w * prod
		}
	}
	if tot > 0 {
		for a := range S {
			S[a] /= tot
		}
	}
	return S
}

// HalfLife is the smallest age with S ≤ 1/2, or len(S) when none.
func HalfLife(S []float64) int {
	for a, s := range S {
		if s <= 0.5 {
			return a
		}
	}
	return len(S)
}

// Capacity is M = Σ_a S(a), the expected number of retrievable items.
func Capacity(S []float64) float64 {
	var m float64
	for _, s := range S {
		m += s
	}
	return m
}

// SurvivalSchedule is Theorem 3: the metric at write j is scale(j)·d₀ and
// write k carries head start lam(k). Item i survives to step t with
//
//	S = (1/n) Σ_s Π_{j=i+1..t} (1 − |B⁰_s((1/scale(j)) Σ_{k=i+1..j} lam(k))| / N)
func SurvivalSchedule(tables []Ball, i, t int, scale, lam func(int) float64) float64 {
	var S float64
	for _, b := range tables {
		prod, acc := 1.0, 0.0
		for j := i + 1; j <= t; j++ {
			acc += lam(j)
			prod *= 1 - b.Open(acc/scale(j))
		}
		S += prod
	}
	if len(tables) == 0 {
		return 0
	}
	return S / float64(len(tables))
}

// ScaledLambda is Λ = c^(1/D)·λ·N.
func ScaledLambda(c, lambda float64, n, d int) float64 {
	return math.Pow(c, 1/float64(d)) * lambda * float64(n)
}

// UniversalSurvival is S_Λ(u) = exp(−u − Λ^D/(D+1)·u^(D+1)), u = a/N.
func UniversalSurvival(u, Lambda float64, d int) float64 {
	D := float64(d)
	return math.Exp(-u - math.Pow(Lambda, D)/(D+1)*math.Pow(u, D+1))
}

// UniversalHalfLife solves x + Λ^D/(D+1)·x^(D+1) = ln 2 for x = W/N.
func UniversalHalfLife(Lambda float64, d int) float64 {
	D := float64(d)
	f := func(x float64) float64 { return x + math.Pow(Lambda, D)/(D+1)*math.Pow(x, D+1) - math.Ln2 }
	lo, hi := 0.0, math.Ln2
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if f(mid) > 0 {
			hi = mid
		} else {
			lo = mid
		}
	}
	return (lo + hi) / 2
}

// UniversalCapacity is M/N = ∫₀^∞ S_Λ(u) du.
func UniversalCapacity(Lambda float64, d int) float64 {
	var s float64
	du := 1e-4
	for u := du / 2; u < 40; u += du {
		v := UniversalSurvival(u, Lambda, d)
		s += v * du
		if v < 1e-12 {
			break
		}
	}
	return s
}

// CapacityAsymptote is Γ(1+1/(D+1))·((D+1)/Λ^D)^(1/(D+1)), the large-Λ M/N.
func CapacityAsymptote(Lambda float64, d int) float64 {
	D := float64(d)
	return math.Gamma(1+1/(D+1)) * math.Pow((D+1)/math.Pow(Lambda, D), 1/(D+1))
}

// DesignLambda is λ* = ((D+1)(N ln 2 − W) / (c·W^(D+1)))^(1/D), the head start
// that keeps the most recent W writes. W ≥ N ln 2 has no λ: 0 is returned.
func DesignLambda(n, w int, c float64, d int) float64 {
	D := float64(d)
	gap := float64(n)*math.Ln2 - float64(w)
	if w < 1 || gap <= 0 || c <= 0 {
		return 0
	}
	return math.Pow((D+1)*gap/(c*math.Pow(float64(w), D+1)), 1/D)
}

// DesignWindow reports whether λ·W lies where the universal law holds:
// 2.5·w_min ≤ λW ≤ 0.15·diam (10·w_min for D = 1).
func DesignWindow(lambda float64, w, wmin, diam, d int) bool {
	lw := lambda * float64(w)
	lo := 2.5 * float64(wmin)
	if d == 1 {
		lo = 10 * float64(wmin)
	}
	return lw >= lo && lw <= 0.15*float64(diam)
}

// DesignLambdaExact solves Theorem 1 for the λ whose half-life is W, by
// bisection on [lo, hi] (the half-life falls as λ grows).
func DesignLambdaExact(tables []Ball, w int, lo, hi float64) float64 {
	ages := 4*w + 16
	for i := 0; i < 60; i++ {
		mid := (lo + hi) / 2
		if HalfLife(Survival(tables, nil, mid, ages)) > w {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2
}

// Regime is the retention regime of Section 4.3.
func Regime(lambda float64, n, w, wmin, diam, d int) string {
	switch {
	case lambda == 0:
		return "primacy (λ = 0: earlier writes win ties)"
	case lambda*float64(n)*math.Ln2 <= float64(wmin):
		return "hash plateau"
	case !DesignWindow(lambda, w, wmin, diam, d) && lambda*float64(w) < 0.15*float64(diam):
		return "transition"
	case lambda*float64(w) > 0.15*float64(diam):
		return "saturation"
	default:
		return "continuum"
	}
}

// OwnProb is Theorem 4: the probability that the item with index i and age a,
// whose source is at distance rho from v, owns v.
//
//	Π_{m=1..a} (1 − p(B_v(ρ+λm))) · Π_{m=1..i−1} (1 − p(B̄_v(ρ−λm)))
func OwnProb(bv Ball, rho, lambda float64, a, i int) float64 {
	prod := 1.0
	for m := 1; m <= a; m++ {
		prod *= 1 - bv.Open(rho+lambda*float64(m))
	}
	for m := 1; m < i; m++ {
		r := rho - lambda*float64(m)
		if r < 0 && lambda > 0 {
			break
		}
		prod *= 1 - bv.Closed(r)
	}
	return prod
}

// BasinArea is Corollary 2(i): the expected number of vertices item i of age
// a owns, averaged over the sampled sources. transitive uses each source's
// own table for every v (exact on a vertex-transitive torus); otherwise every
// vertex's table is built, which costs N shortest-path rows.
func BasinArea(g *Graph, w []int, srcs []int, lambda float64, a, i int, transitive bool) float64 {
	var vt []Ball
	if !transitive {
		vt = make([]Ball, g.N)
		for v := range vt {
			vt[v] = NewBall(g, v, w, nil)
		}
	}
	var tot float64
	for _, s := range srcs {
		ds := g.Distances(s, w)
		bs := NewBall(g, s, w, nil)
		for v := 0; v < g.N; v++ {
			b := bs
			if !transitive {
				b = vt[v]
			}
			tot += OwnProb(b, float64(ds[v]), lambda, a, i)
		}
	}
	if len(srcs) == 0 {
		return 0
	}
	return tot / float64(len(srcs))
}

// NoisyRecall is Corollary 2(ii): owner-recall of item i of age a when the
// query reads v drawn with the source s; pairs are samples of (s, v).
// classes > 0 gives 2(iii), the chance the returned payload is right when
// payloads are one of that many classes.
func NoisyRecall(g *Graph, w []int, pairs [][2]int, lambda float64, a, i, classes int) float64 {
	tabs := map[int]Ball{}
	var r float64
	for _, sv := range pairs {
		s, v := sv[0], sv[1]
		b, ok := tabs[v]
		if !ok {
			b = NewBall(g, v, w, nil)
			tabs[v] = b
		}
		ds := g.Distances(s, w)
		r += OwnProb(b, float64(ds[v]), lambda, a, i)
	}
	if len(pairs) == 0 {
		return 0
	}
	r /= float64(len(pairs))
	if classes > 0 {
		r += (1 - r) / float64(classes)
	}
	return r
}

// ShiftOwn is Proposition 5: owning a vertex at scaled distance r at scaled
// age u is as likely as surviving to u + r/Λ.
func ShiftOwn(u, r, Lambda float64, d int) float64 {
	if Lambda <= 0 {
		return 0
	}
	return UniversalSurvival(u+r/Lambda, Lambda, d)
}

// UniversalBasin is eq. (7): A(a) ≈ S_Λ(u) + DΛ^D ∫_u^∞ (w−u)^(D−1) e^(−w − Λ^D w^(D+1)/(D+1)) dw.
func UniversalBasin(u, Lambda float64, d int) float64 {
	D := float64(d)
	ld := math.Pow(Lambda, D)
	var s float64
	dw := 1e-4
	for w := u + dw/2; w < u+40; w += dw {
		e := math.Exp(-w - ld/(D+1)*math.Pow(w, D+1))
		s += math.Pow(w-u, D-1) * e * dw
		if e < 1e-14 {
			break
		}
	}
	return UniversalSurvival(u, Lambda, d) + D*ld*s
}

// NewestBasinRatio is eq. (9): A(0)/(N/M) → D/(D+1)·π/sin(π/(D+1)).
func NewestBasinRatio(d int) float64 {
	D := float64(d)
	return D / (D + 1) * math.Pi / math.Sin(math.Pi/(D+1))
}

// NoisyLambda is eq. (10): the Λ that maximizes owner-recall for a query
// displaced by scaled distance r at scaled age u. It solves
// (1+y)^D (Dy − 1) = (D+1)/r^D for y > 1/D and returns Λ* = y·r/u.
func NoisyLambda(r, u float64, d int) float64 {
	if r <= 0 || u <= 0 {
		return 0
	}
	D := float64(d)
	rhs := (D + 1) / math.Pow(r, D)
	f := func(y float64) float64 { return math.Pow(1+y, D)*(D*y-1) - rhs }
	lo, hi := 1/D, 1/D+1
	for f(hi) < 0 {
		hi *= 2
	}
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if f(mid) > 0 {
			hi = mid
		} else {
			lo = mid
		}
	}
	return (lo + hi) / 2 * r / u
}

// TokenLocalSurvival is Theorem 5: each write's metric is its own token
// class's. q(s, r) = Pr[d_θ(s', s) < r] over a fresh (source, class) pair:
//
//	S(a) = (1/n) Σ_s Π_{m=1..a} (1 − q(s, λm))
//
// classProb nil is uniform over the classes.
func TokenLocalSurvival(g *Graph, sp *Speeds, classProb []float64, srcs []int, lambda float64, ages int) []float64 {
	S := make([]float64, ages+1)
	k := len(sp.W)
	for _, s := range srcs {
		tabs := make([]Ball, k)
		for c := 0; c < k; c++ {
			tabs[c] = NewBall(g, s, sp.Weights(c), nil)
		}
		q := func(r float64) float64 {
			var t float64
			for c := 0; c < k; c++ {
				pc := 1 / float64(k)
				if classProb != nil {
					pc = classProb[c]
				}
				t += pc * tabs[c].Open(r)
			}
			return t
		}
		prod := 1.0
		S[0]++
		for a := 1; a <= ages; a++ {
			prod *= 1 - q(lambda*float64(a))
			S[a] += prod
		}
	}
	if len(srcs) > 0 {
		for a := range S {
			S[a] /= float64(len(srcs))
		}
	}
	return S
}

// Sandwich is Theorem 6: any metric within κ of the base keeps survival
// between the base law at head starts κλ and λ/κ.
func Sandwich(tables []Ball, lambda, kappa float64, ages int) (lo, hi []float64) {
	return Survival(tables, nil, kappa*lambda, ages), Survival(tables, nil, lambda/kappa, ages)
}

// Anchor is the re-anchoring rule of the retention paper §6.3 for a metric
// that dilates: when Ξ = s/s_a + s_a/s reaches M*, set λ ← λ₀·s/s₀ and
// s_a ← s. With r + 1/r = M* this keeps Λ within [Λ₀/r, Λ₀·r].
type Anchor struct {
	Lambda0, S0, Sa, MStar float64
	Events             int
}

// NewAnchor starts at scale s0 with head start lambda0. mstar ≤ 2 uses 2.08,
// the paper's value (r = 1.33).
func NewAnchor(lambda0, s0, mstar float64) *Anchor {
	if mstar <= 2 {
		mstar = 2.08
	}
	return &Anchor{Lambda0: lambda0, S0: s0, Sa: s0, MStar: mstar}
}

// Update reads the realized scale s and returns the head start to use and
// whether a re-anchoring event fired.
func (an *Anchor) Update(s, current float64) (float64, bool) {
	if s <= 0 || an.Sa <= 0 {
		return current, false
	}
	if s/an.Sa+an.Sa/s < an.MStar {
		return current, false
	}
	an.Sa = s
	an.Events++
	return an.Lambda0 * s / an.S0, true
}
