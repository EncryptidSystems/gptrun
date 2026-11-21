package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"gptrun/internal/hes"
	"gptrun/internal/hes/hesgpu"
)

// hesCmd inspects the holonomy–eikonal state: a saved chat field (-state), or
// a synthetic one of -writes random keys. It prints the laws, the H₁ readout,
// and checks the GPU sweeps against the MRH writer and the CPU soft field.
func hesCmd(args []string) error {
	fs := flag.NewFlagSet("hes", flag.ContinueOnError)
	state := fs.String("state", "", "a state file saved by run -hes")
	dims := fs.String("dims", "64x64", "torus sides")
	writes := fs.Int("writes", 512, "synthetic writes")
	lambda := fs.Int("lambda", -1, "head start per write (-1: design rule from -window)")
	window := fs.Int("window", 512, "retention target W")
	beta := fs.Float64("beta", 0.01, "soft-min β for the GPU soft sweep")
	gpu := fs.Bool("gpu", true, "run the GPU sweeps")
	if err := fs.Parse(args); err != nil {
		return err
	}
	var h *hes.HES
	var err error
	if *state != "" {
		if h, err = hes.LoadHES(*state); err != nil {
			return err
		}
		fmt.Printf("state %s\n", *state)
	} else {
		var d []int
		for _, p := range strings.Split(*dims, "x") {
			var n int
			if _, e := fmt.Sscan(p, &n); e != nil || n < 1 {
				return fmt.Errorf("gptrun hes: -dims %q", *dims)
			}
			d = append(d, n)
		}
		if h, err = hes.NewHES(hes.Options{Dims: d, Seed: 1, Lambda: *lambda, Window: *window, Classes: 1}); err != nil {
			return err
		}
		rng := rand.New(rand.NewSource(2))
		t0 := time.Now()
		for i := 0; i < *writes; i++ {
			key := make([]float32, 64)
			for j := range key {
				key[j] = float32(rng.NormFloat64())
			}
			if _, err := h.Write(hes.Item{IDs: []int{rng.Intn(50000)}, Key: key, Text: fmt.Sprint("item ", i)}); err != nil {
				return err
			}
		}
		fmt.Printf("synthetic  %d writes in %s\n", *writes, time.Since(t0).Round(time.Millisecond))
	}
	r := h.Laws()
	fmt.Printf("field      N %d  D %d  λ %d  writes %d  payloads %d  live %d\n", r.N, r.D, r.Lambda, r.Writes, r.Payloads, r.Live)
	fmt.Printf("growth     c %.4g  exponent %.3f  diam %d  w_min %d\n", r.C, r.Exponent, r.Diam, r.WMin)
	fmt.Printf("retention  Λ %.4g  %s  half-life %d (universal %.1f)  capacity %.1f  sandwich %d–%d  anchors %d\n",
		r.Scaled, r.Regime, r.HalfLifeExact, r.HalfLifeUniversal, r.Capacity, r.SandwichLo, r.SandwichHi, r.Anchors)
	fmt.Printf("basins     newest/mean limit %.3f\n", r.NewestBasinRatio)
	fmt.Printf("holonomy   S_n %s  SO(n) trace %.3f\n", h.Holonomy(), h.Rotation().Trace())
	cls := h.Persistence(h.G.Base, 0)
	obs, ess := 0, 0
	for _, c := range cls {
		if c.Obstruction {
			obs++
		}
		if c.Essential {
			ess++
		}
	}
	fmt.Printf("H1         %d classes (persistence ≥ one edge)  %d essential  %d obstructions\n", len(cls), ess, obs)
	if !*gpu || len(h.F.Payload) == 0 {
		return nil
	}
	var src []int
	var ib []int
	var fb []float64
	var tok [][]int
	for pid := range h.F.Payload {
		if s := h.F.SrcOf[pid]; s >= 0 {
			src, ib, fb = append(src, s), append(ib, h.F.BirthOf[pid]), append(fb, float64(h.F.BirthOf[pid]))
			tok = append(tok, h.Sp.Weights(h.ClassOf(pid)))
		}
	}
	// The MRH writer against the analytic field of the same writes, each under
	// its own token metric: equal with one metric, never below it with many
	// (Proposition 6).
	Ta, _ := hes.AnalyticField(h.G, src, ib, tok)
	off, below := 0, 0
	for v := range Ta {
		if h.F.T[v] != Ta[v] {
			off++
		}
		if h.F.T[v] < Ta[v] {
			below++
		}
	}
	fmt.Printf("writer     MRH field vs analytic: %d vertices differ, %d below it (Proposition 6 allows 0)\n", off, below)
	// The GPU hard sweep against the analytic field under one metric.
	w := h.Sp.Weights(0)
	sw, err := hesgpu.Open(nil, h.G, w)
	if err != nil {
		return fmt.Errorf("gpu: %w", err)
	}
	defer sw.Close()
	t0 := time.Now()
	hard, err := sw.Arrival(src, fb, 0, true, 1<<16, 0.5)
	if err != nil {
		return err
	}
	ws := make([][]int, len(src))
	for i := range ws {
		ws[i] = w
	}
	T, _ := hes.AnalyticField(h.G, src, ib, ws)
	worst := 0.0
	for v := range T {
		worst = math.Max(worst, math.Abs(hard.T[v]-float64(T[v])))
	}
	fmt.Printf("gpu hard   %d sweeps in %s  max |GPU − analytic| %.3g\n", hard.Sweeps, time.Since(t0).Round(time.Millisecond), worst)
	t0 = time.Now()
	soft, err := sw.Arrival(src, fb, *beta, false, 1<<16, 1e-3)
	if err != nil {
		return err
	}
	gap := 0.0
	for v := range T {
		gap = math.Max(gap, float64(T[v])-soft.T[v])
	}
	fmt.Printf("gpu soft   β %g  %d sweeps in %s  hard − soft ≤ %.3g (log of the fronts that meet)/β\n", *beta, soft.Sweeps, time.Since(t0).Round(time.Millisecond), gap)
	return nil
}
