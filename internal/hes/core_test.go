package hes

import "testing"

func TestHolonomySeparatesWhatASumCollides(t *testing.T) {
	// Adjacent transpositions in S5. A diagonal state that adds the generator
	// index cannot tell s0·s1 from s1·s0. The product can.
	s0 := Adjacent(S5, 0)
	s1 := Adjacent(S5, 1)
	if s0.Mul(s1).Eq(s1.Mul(s0)) {
		t.Fatal("s0·s1 commuted")
	}
	if (0+1)%S5 != (1+0)%S5 {
		t.Fatal("the diagonal baseline should collide on this pair")
	}
	word := []Perm{s0, s1, Adjacent(S5, 2), Adjacent(S5, 3), s0}
	g := Identity(S5)
	for _, p := range word {
		g = g.Mul(p)
	}
	swapped := Identity(S5).Mul(s1).Mul(s0).Mul(Adjacent(S5, 2)).Mul(Adjacent(S5, 3)).Mul(s0)
	if g.Eq(swapped) {
		t.Fatal("swapping the first pair did not change the word")
	}
	if !g.Inv().Mul(g).IsIdentity() {
		t.Fatal("g⁻¹g is not the identity")
	}
}

// Chen's identity on the state's S_n channel: developing a prefix and then a
// suffix equals developing the whole list, and equals prefix · suffix.
func TestChen(t *testing.T) {
	mk := func() *HES {
		h, err := NewHES(Options{Dims: []int{4, 4}, Seed: 1, Classes: 1, RotN: 1})
		if err != nil {
			t.Fatal(err)
		}
		return h
	}
	split, whole, prefix, suffix := mk(), mk(), mk(), mk()
	split.Write(Item{IDs: []int{1, 2}})
	split.Write(Item{IDs: []int{3, 4, 5}})
	whole.Write(Item{IDs: []int{1, 2, 3, 4, 5}})
	prefix.Write(Item{IDs: []int{1, 2}})
	suffix.Write(Item{IDs: []int{3, 4, 5}})
	want := whole.Holonomy()
	if !split.Holonomy().Eq(want) || !prefix.Holonomy().Mul(suffix.Holonomy()).Eq(want) {
		t.Fatalf("Chen: split %s, prefix·suffix %s, whole %s", split.Holonomy(), prefix.Holonomy().Mul(suffix.Holonomy()), want)
	}
}

func TestFitKeepsAPrefixInsideTheBudget(t *testing.T) {
	nTok := func(s string) int { return len([]rune(s)) }
	if got := Fit("Question?", "abcdefghij", 40, nTok); got != "abcdefghij" {
		t.Fatalf("full payload %q", got)
	}
	short := Fit("Q", "abcdefghij", 14, nTok)
	if short == "" || short == "abcdefghij" {
		t.Fatalf("prefix %q", short)
	}
	if nTok(Prompt("Q", short))-nTok("Q") > 14 {
		t.Fatalf("prefix %q costs %d", short, nTok(Prompt("Q", short))-nTok("Q"))
	}
	if Fit("Q", "abcdefghij", 0, nTok) != "" {
		t.Fatal("a zero budget returned text")
	}
}

func TestRadixTuplesWithTheSameProductAgree(t *testing.T) {
	keys := []int{3, 0, 7, 1, 5, 2, 4, 6}
	a, err := popOrder(keys, []int{2, 2, 2}, 7)
	if err != nil {
		t.Fatal(err)
	}
	b, err := popOrder(keys, []int{4, 2}, 7)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("order %v vs %v", a, b)
		}
		if i > 0 && keys[a[i]] < keys[a[i-1]] {
			t.Fatalf("key went backwards: %v", a)
		}
	}
}

func TestKeyOutsideTheWindowIsRefused(t *testing.T) {
	h, err := newMRH([]int{2, 2}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.insert(0, 4); err == nil {
		t.Fatal("key 4 was inserted into a window of 3")
	}
}
