package hes

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
)

func torus(t *testing.T, dims []int, w int, seed int64) *Graph {
	t.Helper()
	g, err := Torus(dims, w, 0.2, seed)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

// The incremental MRH writer reproduces T = min_i(b_i + d(s_i, v)) and its
// owners on every vertex (HES note: "matched its analytic formula exactly").
func TestIncrementalMatchesAnalytic(t *testing.T) {
	g := torus(t, []int{16, 16}, 100, 1)
	rng := rand.New(rand.NewSource(2))
	for _, lam := range []int{0, 3, 40} {
		f, _ := NewEik(g, S5)
		var src, birth []int
		for i := 0; i < 300; i++ {
			s := rng.Intn(g.N)
			b := -lam * i
			src, birth = append(src, s), append(birth, b)
			if _, err := f.March(s, b, g.W, fmt.Sprint(i), nil); err != nil {
				t.Fatal(err)
			}
		}
		T, own := AnalyticField(g, src, birth, nil)
		for v := 0; v < g.N; v++ {
			if f.T[v] != T[v] {
				t.Fatalf("λ=%d vertex %d: T %d, analytic %d", lam, v, f.T[v], T[v])
			}
			if f.Payload[f.P[v]] != fmt.Sprint(own[v]) {
				t.Fatalf("λ=%d vertex %d: owner %s, analytic %d", lam, v, f.Payload[f.P[v]], own[v])
			}
		}
	}
}

// Lemma 1: item i keeps its source iff no later write j lands within
// λ(j − i) of it.
func TestLemma1(t *testing.T) {
	g := torus(t, []int{12, 12}, 100, 3)
	rng := rand.New(rand.NewSource(4))
	lam := 7
	f, _ := NewEik(g, S5)
	var src []int
	for i := 0; i < 400; i++ {
		s := rng.Intn(g.N)
		src = append(src, s)
		f.March(s, -lam*i, g.W, fmt.Sprint(i), nil)
	}
	for i, s := range src {
		d := g.Distances(s, nil)
		alive := true
		for j := i + 1; j < len(src); j++ {
			if d[src[j]] < lam*(j-i) {
				alive = false
				break
			}
		}
		if got := f.OwnsSource(fmt.Sprint(i)); got != alive {
			t.Fatalf("item %d: owns %v, Lemma 1 says %v", i, got, alive)
		}
	}
}

// Proposition 6: with token metrics, every item alive in the analytic field is
// alive under the pruned writer.
func TestPrunedWriterKeepsAnalyticSurvivors(t *testing.T) {
	g := torus(t, []int{16, 16}, 1000, 5)
	sp := NewSpeeds(g, 8, 2, 4, 6)
	rng := rand.New(rand.NewSource(7))
	lam := 300
	f, _ := NewEik(g, S5)
	var src, birth []int
	var ws [][]int
	for i := 0; i < 500; i++ {
		s, c := rng.Intn(g.N), rng.Intn(8)
		src, birth, ws = append(src, s), append(birth, -lam*i), append(ws, sp.Weights(c))
		f.March(s, -lam*i, sp.Weights(c), fmt.Sprint(i), nil)
	}
	_, own := AnalyticField(g, src, birth, ws)
	for i, s := range src {
		if own[s] == i && !f.OwnsSource(fmt.Sprint(i)) {
			t.Fatalf("item %d alive in the analytic field, evicted by the pruned writer", i)
		}
	}
}

// Corollary 1: on the plateau the half-life is N ln 2 (a last-write-wins table).
func TestPlateauHalfLife(t *testing.T) {
	g := torus(t, []int{16, 16}, 1000, 8)
	var tabs []Ball
	for _, s := range []int{0, 37, 101, 200} {
		tabs = append(tabs, NewBall(g, s, nil, nil))
	}
	lam := 1.0 // λ·N·ln2 = 177 ≤ w_min = 800
	S := Survival(tabs, nil, lam, 400)
	want := math.Log(2) / -math.Log(1-1/float64(g.N))
	if got := HalfLife(S); math.Abs(float64(got)-want) > 1.5 {
		t.Fatalf("plateau half-life %d, want %.1f", got, want)
	}
	if r := Regime(lam, g.N, HalfLife(S), g.WMin(), 40000, 2); r != "hash plateau" {
		t.Fatalf("regime %q", r)
	}
}

func TestUniversalLaw(t *testing.T) {
	if x := UniversalHalfLife(1e-6, 2); math.Abs(x-math.Ln2) > 1e-6 {
		t.Fatalf("Λ→0: %g", x)
	}
	// Large Λ: x ≈ ((D+1) ln2)^(1/(D+1)) Λ^(−D/(D+1)).
	L := 1e4
	want := math.Pow(3*math.Ln2, 1.0/3) * math.Pow(L, -2.0/3)
	if x := UniversalHalfLife(L, 2); math.Abs(x/want-1) > 0.01 {
		t.Fatalf("Λ=1e4: %g, asymptote %g", x, want)
	}
	if r := NewestBasinRatio(1); math.Abs(r-math.Pi/2) > 1e-12 {
		t.Fatalf("D=1 ratio %g", r)
	}
	if r := NewestBasinRatio(2); math.Abs(r-2.418) > 1e-3 {
		t.Fatalf("D=2 ratio %g", r)
	}
	if m, a := UniversalCapacity(1000, 2), CapacityAsymptote(1000, 2); math.Abs(m/a-1) > 0.07 {
		t.Fatalf("capacity %g, asymptote %g", m, a)
	}
	y := NoisyLambda(2, 0.01, 2) * 0.01 / 2
	if lhs := math.Pow(1+y, 2) * (2*y - 1); math.Abs(lhs-3.0/4) > 1e-9 {
		t.Fatalf("eq. (10) residual %g", lhs-0.75)
	}
}

// The design rule lands near its target window on the prototype's grid size.
func TestDesignLambda(t *testing.T) {
	g := torus(t, []int{32, 32}, 1000, 9)
	gr := MeasureGrowth(g, nil, 24, 10)
	if math.Abs(gr.Exponent-2) > 0.35 {
		t.Fatalf("growth exponent %g on a 2-torus", gr.Exponent)
	}
	lam := DesignLambda(g.N, 100, gr.C, 2)
	var tabs []Ball
	for _, s := range []int{1, 300, 600, 900, 1000, 77} {
		tabs = append(tabs, NewBall(g, s, nil, nil))
	}
	w := HalfLife(Survival(tabs, nil, lam, 2000))
	if w < 70 || w > 130 {
		t.Fatalf("λ*=%.1f gives half-life %d, target 100", lam, w)
	}
	ex := DesignLambdaExact(tabs, 100, 0, float64(gr.Diam))
	if w := HalfLife(Survival(tabs, nil, ex, 2000)); w < 97 || w > 103 {
		t.Fatalf("exact λ=%.1f gives half-life %d", ex, w)
	}
}

// Theorem 1 agrees with the mechanism: simulated survival by age matches.
func TestSurvivalMatchesMechanism(t *testing.T) {
	g := torus(t, []int{16, 16}, 100, 12)
	lam := 20
	var tabs []Ball
	for s := 0; s < g.N; s += 8 {
		tabs = append(tabs, NewBall(g, s, nil, nil))
	}
	S := Survival(tabs, nil, float64(lam), 300)
	alive := make([]float64, 301)
	count := make([]float64, 301)
	rng := rand.New(rand.NewSource(13))
	for run := 0; run < 6; run++ {
		f, _ := NewEik(g, S5)
		K := 3 * g.N
		for i := 0; i < K; i++ {
			f.March(rng.Intn(g.N), -lam*i, g.W, fmt.Sprint(i), nil)
		}
		for a := 0; a < 300; a++ {
			i := K - 1 - a
			count[a]++
			if f.OwnsSource(fmt.Sprint(i)) {
				alive[a]++
			}
		}
	}
	// 30-age bins of 180 items each: a binomial standard error near 0.037.
	worst := 0.0
	for lo := 0; lo < 300; lo += 30 {
		var sim, th, n float64
		for a := lo; a < lo+30; a++ {
			sim += alive[a]
			th += S[a] * count[a]
			n += count[a]
		}
		worst = math.Max(worst, math.Abs(sim/n-th/n))
	}
	if worst > 0.11 {
		t.Fatalf("largest bin |sim − Theorem 1| = %.3f", worst)
	}
	t.Logf("largest bin |sim − Theorem 1| = %.3f", worst)
}

// A torus is filled by its cells except its two essential loops.
func TestPersistenceTorusRank(t *testing.T) {
	g := torus(t, []int{10, 10}, 100, 14)
	f, _ := NewEik(g, S5)
	f.March(0, 0, g.W, "a", nil)
	f.March(55, 0, g.W, "b", nil)
	cls := f.Persistence(0, 0)
	ess := 0
	for _, c := range cls {
		if c.Essential {
			ess++
		}
		if len(c.Cycle) < 3 {
			t.Fatalf("class without a cycle: %+v", c)
		}
	}
	if ess != 2 {
		t.Fatalf("%d essential classes, a torus has 2", ess)
	}
	// Two fronts meeting: the defect across the boundary is recorded.
	obs := 0
	for _, c := range cls {
		if c.Obstruction {
			obs++
		}
	}
	t.Logf("%d classes, %d essential, %d obstructions", len(cls), ess, obs)
}

func TestSoftApproachesHard(t *testing.T) {
	g := torus(t, []int{8, 8}, 100, 15)
	src := []int{0, 27}
	birth := []float64{0, -50}
	sf := SoftArrival(g, nil, src, birth, 1.0, 5000, 1e-9)
	T, _ := AnalyticField(g, src, []int{0, -50}, nil)
	for v := range T {
		if d := float64(T[v]) - sf.T[v]; d < -1e-9 || d > 10 {
			t.Fatalf("vertex %d: hard %d soft %g", v, T[v], sf.T[v])
		}
	}
	// Gradient against finite differences, at a temperature where it is smooth.
	beta := 0.05
	w := append([]int(nil), g.W...)
	dL := make([]float64, g.N)
	for v := range dL {
		dL[v] = float64(v%5) - 2
	}
	loss := func(w []int) float64 {
		s := SoftArrival(g, w, src, birth, beta, 20000, 1e-12)
		var l float64
		for v := range s.T {
			l += dL[v] * s.T[v]
		}
		return l
	}
	sf = SoftArrival(g, w, src, birth, beta, 20000, 1e-12)
	dW, _ := SoftGrad(g, w, sf, dL, 20000, 1e-12)
	for _, e := range []int{0, 5, 17, 40} {
		wp := append([]int(nil), w...)
		wm := append([]int(nil), w...)
		wp[e]++
		wm[e]--
		fd := (loss(wp) - loss(wm)) / 2
		if math.Abs(fd-dW[e]) > 1e-3*math.Max(1, math.Abs(fd)) {
			t.Fatalf("edge %d: adjoint %g, finite difference %g", e, dW[e], fd)
		}
	}
}

func TestHouseholderChen(t *testing.T) {
	rng := rand.New(rand.NewSource(16))
	vec := func() []float64 {
		v := make([]float64, 6)
		for i := range v {
			v[i] = rng.NormFloat64()
		}
		return v
	}
	a := Householder2(6, vec(), vec())
	b := Householder2(6, vec(), vec())
	c := Householder2(6, vec(), vec())
	if d := a.Mul(b).Mul(c).Distance(a.Mul(b.Mul(c))); d > 1e-12 {
		t.Fatalf("Chen: %g", d)
	}
	if a.Mul(b).Distance(b.Mul(a)) < 1e-6 {
		t.Fatal("the orthogonal channel commutes")
	}
	// Orthogonal: rᵀr = I.
	ab := a.Mul(b)
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			var s float64
			for k := 0; k < 6; k++ {
				s += ab.A[k*6+i] * ab.A[k*6+j]
			}
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(s-want) > 1e-12 {
				t.Fatalf("not orthogonal at %d,%d: %g", i, j, s)
			}
		}
	}
}

func randKey(rng *rand.Rand, m int) []float32 {
	k := make([]float32, m)
	var ss float64
	for i := range k {
		k[i] = float32(rng.NormFloat64())
		ss += float64(k[i]) * float64(k[i])
	}
	for i := range k {
		k[i] /= float32(math.Sqrt(ss))
	}
	return k
}

func noisy(rng *rand.Rand, k []float32, sigma float64) []float32 {
	out := make([]float32, len(k))
	var ss float64
	for i := range k {
		out[i] = k[i] + float32(sigma*rng.NormFloat64()/math.Sqrt(float64(len(k))))
		ss += float64(out[i]) * float64(out[i])
	}
	for i := range out {
		out[i] /= float32(math.Sqrt(ss))
	}
	return out
}

// A noisy query still reaches its item's basin far more often than a hash
// of the query would (HES note, Noisy queries).
func TestFuzzyRecall(t *testing.T) {
	h, err := NewHES(Options{Dims: []int{32, 32}, Seed: 17, Classes: 1, RotN: 1})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(18))
	var keys [][]float32
	for i := 0; i < 64; i++ {
		keys = append(keys, randKey(rng, 32))
	}
	h.FitKeys(keys)
	for i, k := range keys {
		if _, err := h.Write(Item{IDs: []int{i + 1}, Key: k, Text: fmt.Sprint(i)}); err != nil {
			t.Fatal(err)
		}
	}
	hit := 0
	for i, k := range keys {
		top := h.ReadAll(noisy(rng, k, 0.1), nil, nil, 3)
		for _, r := range top {
			if r.Payload == fmt.Sprint(i) {
				hit++
				break
			}
		}
	}
	if hit < 40 {
		t.Fatalf("noisy recall in the top 3: %d of 64", hit)
	}
	t.Logf("noisy recall in the top 3: %d of 64", hit)
}

// The learned φ keeps what the keys know: with noisy queries the item read
// first is the item asked for about as often as a cosine ranking finds it.
func TestLearnedPhiRecall(t *testing.T) {
	rng := rand.New(rand.NewSource(40))
	var keys [][]float32
	for i := 0; i < 200; i++ {
		keys = append(keys, randKey(rng, 128))
	}
	run := func(noLearn bool) int {
		h, _ := NewHES(Options{Dims: []int{64, 64}, Seed: 41, NoLearn: noLearn})
		h.FitKeys(keys)
		for i, k := range keys {
			h.Write(Item{IDs: []int{i + 1}, Key: k, Text: fmt.Sprint(i)})
		}
		hit := 0
		qr := rand.New(rand.NewSource(42))
		for i, k := range keys {
			if r, ok := h.Read(noisy(qr, k, 0.5), nil, nil); ok && r.Payload == fmt.Sprint(i) {
				hit++
			}
		}
		return hit
	}
	learned, fixed := run(false), run(true)
	t.Logf("top-1 of 200 at σ=0.5: learned φ %d, projection φ %d", learned, fixed)
	if learned < 180 || learned <= fixed {
		t.Fatalf("learned φ %d, projection %d", learned, fixed)
	}
}

func TestReadOwnerFirstAndCharacteristic(t *testing.T) {
	g := torus(t, []int{10, 10}, 100, 19)
	f, _ := NewEik(g, S5)
	f.March(3, 0, g.W, "a", nil)
	f.March(70, 0, g.W, "b", nil)
	all := f.ReadAll(4, nil, 0)
	if len(all) != 2 || all[0].Payload != "a" || !all[0].Owner || all[0].Basin != 3 {
		t.Fatalf("read at 4: %+v", all)
	}
	r, ok := f.Read(4, func(p string) bool { return p == "a" })
	if !ok || r.Payload != "b" || r.Basin != 70 {
		t.Fatalf("skip a: %+v", r)
	}
}

// A read's basin lies in its own payload's territory, even after later writes
// have taken parents off the characteristic.
func TestBasinStaysInItsPayload(t *testing.T) {
	g := torus(t, []int{16, 16}, 100, 30)
	sp := NewSpeeds(g, 8, 2, 4, 31)
	f, _ := NewEik(g, S5)
	rng := rand.New(rand.NewSource(32))
	for i := 0; i < 200; i++ {
		f.March(rng.Intn(g.N), -5*i, sp.Weights(rng.Intn(8)), fmt.Sprint(i), nil)
	}
	for q := 0; q < g.N; q += 7 {
		for _, r := range f.ReadAll(q, nil, 0) {
			if f.Payload[f.P[r.Basin]] != r.Payload {
				t.Fatalf("query %d: %q has basin %d owned by %q", q, r.Payload, r.Basin, f.Payload[f.P[r.Basin]])
			}
		}
	}
}

func TestSaveLoad(t *testing.T) {
	h, err := NewHES(Options{Dims: []int{16, 16}, Seed: 20, Lambda: -1, Window: 50})
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(21))
	for i := 0; i < 40; i++ {
		h.Write(Item{IDs: []int{i, i + 1}, Key: randKey(rng, 16), Text: fmt.Sprint("p", i)})
	}
	path := filepath.Join(t.TempDir(), "s.hes")
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}
	h2, err := LoadHES(path)
	if err != nil {
		t.Fatal(err)
	}
	q := randKey(rng, 16)
	a, b := h.ReadAll(q, nil, nil, 5), h2.ReadAll(q, nil, nil, 5)
	if len(a) != len(b) {
		t.Fatalf("%d vs %d recalls", len(a), len(b))
	}
	for i := range a {
		if a[i].Payload != b[i].Payload || a[i].Arrival != b[i].Arrival {
			t.Fatalf("recall %d: %+v vs %+v", i, a[i], b[i])
		}
	}
	if h2.Lambda() != h.Lambda() || h2.Writes() != h.Writes() || !h2.Holonomy().Eq(h.Holonomy()) {
		t.Fatal("channels differ after load")
	}
	// Writing continues identically.
	k := randKey(rng, 16)
	i1, _ := h.Write(Item{IDs: []int{9}, Key: k, Text: "next"})
	i2, _ := h2.Write(Item{IDs: []int{9}, Key: k, Text: "next"})
	if i1.Took != i2.Took || i1.Birth != i2.Birth || i1.Vertex != i2.Vertex {
		t.Fatalf("write after load: %+v vs %+v", i1, i2)
	}
}

func TestAnchorKeepsScaledLambda(t *testing.T) {
	an := NewAnchor(100, 1, 2.08)
	lam := 100.0
	for s := 1.0; s > 0.06; s *= 0.97 {
		lam, _ = an.Update(s, lam)
		if r := (lam / s) / 100; r < 1/1.34 || r > 1.34 {
			t.Fatalf("scale %.3f: Λ ratio %.3f outside the band", s, r)
		}
	}
	if an.Events == 0 {
		t.Fatal("no re-anchoring on a contraction")
	}
}

func TestTokenLocalWithinSandwich(t *testing.T) {
	g := torus(t, []int{16, 16}, 1000, 22)
	sp := NewSpeeds(g, 4, 2, 3, 23)
	srcs := []int{0, 50, 120, 200}
	lam := 400.0
	S := TokenLocalSurvival(g, sp, nil, srcs, lam, 300)
	var tabs []Ball
	for _, s := range srcs {
		tabs = append(tabs, NewBall(g, s, nil, nil))
	}
	lo, hi := Sandwich(tabs, lam, 2, 300)
	for a := range S {
		if S[a] < lo[a]-1e-9 || S[a] > hi[a]+1e-9 {
			t.Fatalf("age %d: %g outside [%g, %g]", a, S[a], lo[a], hi[a])
		}
	}
}
