package hodge

import (
	"math"
	"math/rand"
	"testing"
)

func TestDistDropsFarTail(t *testing.T) {
	lg := make([]float32, 8)
	for i := range lg {
		lg[i] = -100
	}
	lg[3] = 0
	lg[5] = -11
	lg[6] = -12
	lg[7] = -13
	got := map[int]bool{}
	for _, c := range dist(lg, 0.6) {
		got[c.id] = true
	}
	if !got[3] || !got[5] || !got[6] || got[7] || got[0] {
		t.Fatalf("window %v", got)
	}
}

func TestRepeatPenaltyArgmax(t *testing.T) {
	lg := []float32{0, 5, 4.9}
	id := draw(lg, []int{1}, 0, 0.9, nil, nil, nil)
	if id != 2 {
		t.Fatalf("id %d logits %v", id, lg)
	}
}

func TestPenaltySkipsStop(t *testing.T) {
	lg := []float32{0, 5, 4}
	id := draw(lg, []int{1}, 0, 0.9, func(id int) bool { return id == 1 }, nil, nil)
	if id != 1 {
		t.Fatalf("id %d logits %v", id, lg)
	}
}

// TestChooseMatchesPick: the fast draw is the reference draw, token for token,
// on the same random stream.
func TestChooseMatchesPick(t *testing.T) {
	src := rand.New(rand.NewSource(7))
	for trial := 0; trial < 400; trial++ {
		n := 1 + src.Intn(5000)
		lg := make([]float32, n)
		spread := []float64{0.5, 3, 10, 40}[trial%4]
		for i := range lg {
			lg[i] = float32(src.NormFloat64() * spread)
		}
		if trial%7 == 0 {
			lg[src.Intn(n)] = float32(math.NaN())
		}
		if trial%11 == 0 {
			lg[src.Intn(n)] = lg[src.Intn(n)] // a tie
		}
		ends := []int{src.Intn(n), src.Intn(n)}
		end := func(id int) bool { return id == ends[0] || id == ends[1] }
		for _, temp := range []float64{0, 0.3, 0.6, 1, 1.5} {
			for _, top := range []float64{0, 0.5, 0.9, 1} {
				seed := src.Int63()
				var r1, r2 *rand.Rand
				if temp > 0 {
					r1, r2 = rand.New(rand.NewSource(seed)), rand.New(rand.NewSource(seed))
				}
				want := pick(dist(lg, temp), temp, top, minP, end, r1)
				got := choose(lg, temp, top, minP, ends, r2)
				if got != want {
					t.Fatalf("trial %d temp %v top %v: choose %d, pick %d", trial, temp, top, got, want)
				}
			}
		}
	}
}

func BenchmarkChooseFlat(b *testing.B) {
	lg := make([]float32, 563206)
	src := rand.New(rand.NewSource(1))
	for i := range lg {
		lg[i] = float32(src.NormFloat64())
	}
	b.Run("choose", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			choose(lg, 0.6, 0.9, minP, []int{5}, rand.New(rand.NewSource(1)))
		}
	})
	b.Run("reference", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			pick(dist(lg, 0.6), 0.6, 0.9, minP, func(id int) bool { return id == 5 }, rand.New(rand.NewSource(1)))
		}
	})
}
