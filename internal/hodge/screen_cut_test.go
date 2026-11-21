package hodge

import (
	"math"
	"testing"
)

func TestWiderCutUsesTheRescoredBest(t *testing.T) {
	bestLo := 10.0
	cut1 := cutUnder(bestLo)
	// The int8 score sits about one bound-width under the real best.
	if cut2, wider := widerCut(cut1, bestLo+20, 0.6); wider {
		t.Fatalf("window %v should sit inside the argmax cut %v", cut2, cut1)
	}
	cut2, wider := widerCut(cut1, bestLo, 0.6)
	if !wider || !(cut2 < cut1) {
		t.Fatalf("a tight bound still has a temperature tail: cut %v argmax %v", cut2, cut1)
	}
	if _, wider := widerCut(cut1, bestLo+20, 0); wider {
		t.Fatal("greedy has no temperature tail")
	}
}

func TestWindowBestAppliesThePenalty(t *testing.T) {
	lg := []float32{1, 4, float32(math.Inf(-1))}
	pen := []bool{false, true, false}
	// 4 / 1.08 is still above 1.
	if got := windowBest(lg, pen); got <= 1 {
		t.Fatalf("penalized best %v", got)
	}
	lg[1] = 1.05
	if got := windowBest(lg, pen); got != 1 {
		t.Fatalf("penalty should hand the best to row 0, got %v", got)
	}
}
