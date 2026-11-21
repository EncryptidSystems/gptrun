package main

import (
	"path/filepath"
	"testing"
)

func TestPublicSetsLoad(t *testing.T) {
	root := filepath.Join("..", "..", "eval", "public")
	want := map[string]int{
		"arc-challenge": 1172,
		"arc-easy":      2376,
		"openbookqa":    500,
	}
	for name, n := range want {
		items, err := loadTask(root, name)
		if err != nil {
			t.Fatal(name, err)
		}
		if len(items) != n {
			t.Fatalf("%s: got %d items, want %d", name, len(items), n)
		}
		if items[0].Gold < 0 || items[0].Gold >= len(items[0].Choices) {
			t.Fatalf("%s: gold %d", name, items[0].Gold)
		}
	}
}

func TestGoldIndex(t *testing.T) {
	labels := []string{"A", "B", "C", "D"}
	if goldIndex("B", labels) != 1 {
		t.Fatal("letter")
	}
	if goldIndex("3", labels) != 2 {
		t.Fatal("number")
	}
	if goldIndex("E", labels) != -1 {
		t.Fatal("missing")
	}
}
