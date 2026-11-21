package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHasWord(t *testing.T) {
	for _, c := range []struct {
		text, alts string
		want       bool
	}{
		{"The capital is Canberra.", "Canberra", true},
		{"the capital is canberra", "Canberra", true},
		{"This is it.", "hi", false}, // not inside "This"
		{"It was 1989.", "1989", true},
		{"In 19890 BC", "1989", false},
		{"Leonardo painted it.", "Leonardo|da Vinci", true},
		{"Gabriel García Márquez wrote it.", "Márquez|Marquez", true},
		{"The Patuxka River", "Patuxka", true},
		{"The Patuxent River", "Patuxka", false},
		{"", "x", false},
	} {
		if got := hasWord(c.text, c.alts); got != c.want {
			t.Errorf("hasWord(%q, %q) = %v", c.text, c.alts, got)
		}
	}
}

func TestScoreAnswer(t *testing.T) {
	missing, found := scoreAnswer("Rudge founded Columbia.", []string{"Rouse"}, []string{"Rudge"})
	if len(missing) != 1 || len(found) != 1 {
		t.Fatalf("%v %v", missing, found)
	}
}

func TestReadEvalSet(t *testing.T) {
	// The shipped set parses.
	cases, err := readEvalSet(filepath.Join("..", "..", "eval", "answers.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	if len(cases) < 30 {
		t.Fatalf("%d cases", len(cases))
	}
	dir := t.TempDir()
	for _, bad := range []string{
		`{"id":"a","q":"x","route":"nope"}`,
		`{"id":"a","q":"x","wants":["typo"]}`,
		`{"id":"a","q":"x"}` + "\n" + `{"id":"a","q":"y"}`,
		`{"q":"x"}`,
	} {
		p := filepath.Join(dir, "bad.jsonl")
		if err := os.WriteFile(p, []byte(bad), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := readEvalSet(p); err == nil {
			t.Errorf("accepted %q", strings.ReplaceAll(bad, "\n", " / "))
		}
	}
}
