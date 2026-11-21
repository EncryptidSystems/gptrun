package hes

import "math"

// A hard min sends gradient only along the winning geodesic. The HES note
// trains with a soft-min (log-sum-exp at temperature 1/β) and anneals β
// upward; it also names fast-sweeping / Bellman–Ford iterations as the form
// that runs per chunk on parallel hardware. This is that form:
//
//	T_β(v) = softmin_β( S(v), min_u T_β(u) + w_uv ),
//	softmin_β(x…) = −(1/β) log Σ e^(−β x),
//
// iterated as Jacobi sweeps (every vertex from the previous sweep, so a
// sweep is one data-parallel pass). As β → ∞ it is the hard field.

// SoftField is a converged soft arrival field.
type SoftField struct {
	Beta   float64
	T      []float64
	S      []float64 // the sources' own softmin birth at each vertex (+Inf elsewhere)
	Sweeps int
	Delta  float64 // largest change in the last sweep
}

func softmin2(beta, a, b float64) float64 {
	if math.IsInf(a, 1) {
		return b
	}
	if math.IsInf(b, 1) {
		return a
	}
	m := math.Min(a, b)
	return m - math.Log(math.Exp(-beta*(a-m))+math.Exp(-beta*(b-m)))/beta
}

// SoftArrival sweeps until the largest change is below tol or maxSweeps.
// Sources are vertices with birth keys; w nil is the base metric. Sweeps
// converge when e^(−β·w_min)·degree < 1.
func SoftArrival(g *Graph, w []int, src []int, birth []float64, beta float64, maxSweeps int, tol float64) SoftField {
	if w == nil {
		w = g.W
	}
	inf := math.Inf(1)
	S := make([]float64, g.N)
	for v := range S {
		S[v] = inf
	}
	for i, s := range src {
		S[s] = softmin2(beta, S[s], birth[i])
	}
	T := append([]float64(nil), S...)
	next := make([]float64, g.N)
	sf := SoftField{Beta: beta, S: S}
	for it := 0; it < maxSweeps; it++ {
		var delta float64
		for v := 0; v < g.N; v++ {
			x := S[v]
			for i := g.off[v]; i < g.off[v+1]; i++ {
				u := g.nbr[i]
				if math.IsInf(T[u], 1) {
					continue
				}
				x = softmin2(beta, x, T[u]+float64(w[g.eid[i]]))
			}
			next[v] = x
			if !math.IsInf(x, 1) {
				d := math.Abs(x - T[v])
				if math.IsInf(T[v], 1) {
					d = math.Inf(1)
				}
				delta = math.Max(delta, d)
			}
		}
		T, next = next, T
		sf.Sweeps = it + 1
		sf.Delta = delta
		if delta < tol {
			break
		}
	}
	sf.T = T
	return sf
}

// SoftGrad is the exact gradient of L = Σ_v dL[v]·T_β(v) at the fixed point,
// with respect to every edge weight and every source birth: the adjoint of
// the soft Bellman–Ford map. With the soft parent shares
//
//	π(v←u) = e^(−β(T(u)+w_uv−T(v))),   π(v←src) = e^(−β(S(v)−T(v))),
//
// the adjoint solves μ_v = dL[v] + Σ_x π(x←v) μ_x, and then
// ∂L/∂w_uv = μ_v π(v←u) + μ_u π(u←v), ∂L/∂S(v) = μ_v π(v←src).
func SoftGrad(g *Graph, w []int, sf SoftField, dL []float64, maxSweeps int, tol float64) (dW, dS []float64) {
	if w == nil {
		w = g.W
	}
	beta, T, S := sf.Beta, sf.T, sf.S
	share := func(from, to float64) float64 {
		if math.IsInf(from, 1) || math.IsInf(to, 1) {
			return 0
		}
		return math.Exp(-beta * (from - to))
	}
	mu := append([]float64(nil), dL...)
	next := make([]float64, g.N)
	for it := 0; it < maxSweeps; it++ {
		var delta float64
		for v := 0; v < g.N; v++ {
			x := dL[v]
			for i := g.off[v]; i < g.off[v+1]; i++ {
				u := g.nbr[i] // v feeds u through edge (v, u)
				x += share(T[v]+float64(w[g.eid[i]]), T[u]) * mu[u]
			}
			next[v] = x
			delta = math.Max(delta, math.Abs(x-mu[v]))
		}
		mu, next = next, mu
		if delta < tol {
			break
		}
	}
	dW = make([]float64, len(g.Edges))
	for e, uv := range g.Edges {
		u, v := uv[0], uv[1]
		we := float64(w[e])
		dW[e] = mu[v]*share(T[u]+we, T[v]) + mu[u]*share(T[v]+we, T[u])
	}
	dS = make([]float64, g.N)
	for v := range dS {
		dS[v] = mu[v] * share(S[v], T[v])
	}
	return dW, dS
}

// SoftOwnership is the soft owner share of each write at each vertex:
// π_i(v) = softmax_i(−β·T_i(v)) with T_i write i's own field b_i + d_i(s_i, v).
// ws[i] nil is the base metric. As β → ∞ it is the hard tessellation.
func SoftOwnership(g *Graph, src []int, birth []float64, ws [][]int, beta float64) [][]float64 {
	k := len(src)
	fields := make([][]int, k)
	for i := range src {
		var w []int
		if i < len(ws) {
			w = ws[i]
		}
		fields[i] = g.Distances(src[i], w)
	}
	own := make([][]float64, k)
	for i := range own {
		own[i] = make([]float64, g.N)
	}
	for v := 0; v < g.N; v++ {
		m := math.Inf(1)
		for i := 0; i < k; i++ {
			if fields[i][v] != noTime {
				m = math.Min(m, birth[i]+float64(fields[i][v]))
			}
		}
		var z float64
		for i := 0; i < k; i++ {
			if fields[i][v] == noTime {
				continue
			}
			e := math.Exp(-beta * (birth[i] + float64(fields[i][v]) - m))
			own[i][v] = e
			z += e
		}
		if z > 0 {
			for i := 0; i < k; i++ {
				own[i][v] /= z
			}
		}
	}
	return own
}

// BetaSchedule is the geometric anneal from beta0 up to beta1 over steps.
func BetaSchedule(beta0, beta1 float64, steps int) []float64 {
	if steps < 2 {
		return []float64{beta1}
	}
	out := make([]float64, steps)
	for i := range out {
		out[i] = beta0 * math.Pow(beta1/beta0, float64(i)/float64(steps-1))
	}
	return out
}
