package hesgpu

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"gptrun/internal/hes"
	"gptrun/internal/rt"
)

func open(t *testing.T, g *hes.Graph, w []int) *Sweeper {
	t.Helper()
	if _, err := os.Stat(rt.Exe()); err != nil {
		t.Skipf("no dawnrt: %v", err)
	}
	s, err := Open(nil, g, w)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	return s
}

// The hard sweep is the Johnson–Mehl field T = min_i(b_i + d(s_i, v)), the
// same values the MRH writer stores.
func TestHardSweepIsTheAnalyticField(t *testing.T) {
	g, _ := hes.Torus([]int{32, 32}, 1000, 0.2, 1)
	sp := hes.NewSpeeds(g, 4, 2, 4, 2)
	w := sp.Weights(2)
	s := open(t, g, w)
	rng := rand.New(rand.NewSource(3))
	var src, ib []int
	var fb []float64
	for i := 0; i < 40; i++ {
		src = append(src, rng.Intn(g.N))
		ib = append(ib, -37*i)
		fb = append(fb, float64(-37*i))
	}
	sf, err := s.Arrival(src, fb, 0, true, 4096, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	var ws [][]int
	for range src {
		ws = append(ws, w)
	}
	T, _ := hes.AnalyticField(g, src, ib, ws)
	for v := range T {
		if math.Abs(sf.T[v]-float64(T[v])) > 0.05 {
			t.Fatalf("vertex %d: GPU %g, analytic %d (after %d sweeps)", v, sf.T[v], T[v], sf.Sweeps)
		}
	}
	t.Logf("hard field on %d vertices in %d sweeps", g.N, sf.Sweeps)
}

func TestSoftSweepAndGradientMatchTheCPU(t *testing.T) {
	g, _ := hes.Torus([]int{16, 16}, 100, 0.2, 4)
	s := open(t, g, nil)
	src := []int{0, 77, 200}
	birth := []float64{0, -40, 15}
	beta := 0.05
	gpu, err := s.Arrival(src, birth, beta, false, 20000, 1e-4)
	if err != nil {
		t.Fatal(err)
	}
	cpu := hes.SoftArrival(g, nil, src, birth, beta, 20000, 1e-9)
	for v := range cpu.T {
		if d := math.Abs(gpu.T[v] - cpu.T[v]); d > 1e-3*math.Max(1, math.Abs(cpu.T[v])) {
			t.Fatalf("vertex %d: GPU %g, CPU %g", v, gpu.T[v], cpu.T[v])
		}
	}
	dL := make([]float64, g.N)
	for v := range dL {
		dL[v] = float64(v%7) - 3
	}
	dWg, dSg, err := s.Grad(gpu, dL, 20000, 1e-5)
	if err != nil {
		t.Fatal(err)
	}
	dWc, dSc := hes.SoftGrad(g, nil, cpu, dL, 20000, 1e-12)
	for e := range dWc {
		if d := math.Abs(dWg[e] - dWc[e]); d > 2e-3*math.Max(1, math.Abs(dWc[e])) {
			t.Fatalf("edge %d: GPU %g, CPU %g", e, dWg[e], dWc[e])
		}
	}
	for v := range dSc {
		if d := math.Abs(dSg[v] - dSc[v]); d > 2e-3*math.Max(1, math.Abs(dSc[v])) {
			t.Fatalf("source %d: GPU %g, CPU %g", v, dSg[v], dSc[v])
		}
	}
	t.Logf("soft field %d sweeps, gradient matches on %d edges", gpu.Sweeps, len(dWc))
}
