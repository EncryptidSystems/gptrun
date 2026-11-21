package spell

import (
	"os"
	"strings"
	"testing"
)

func testBook() *Book {
	b := &Book{
		rank: map[string]int{},
		hold: map[string]bool{"chillin": true},
		dels: map[string][]uint32{},
	}
	for w, r := range map[string]int{
		"the":        10,
		"ten":        500,
		"password":   100,
		"receive":    120,
		"cheesecake": 1000,
		"france":     200,
		"weather":    400,
		"whether":    150,
		"paris":      300,
		"recieve":    90000,
		"teh":        80000,
	} {
		b.rank[w] = r
		if r < candRank {
			id := uint32(len(b.word))
			b.word = append(b.word, w)
			for _, d := range deletes(w) {
				b.dels[d] = append(b.dels[d], id)
			}
		}
	}
	return b
}

func TestTypos(t *testing.T) {
	b := testBook()
	cases := []struct{ in, want string }{
		{"teh", "the"},
		{"pasword", "password"},
		{"recieve", "receive"},
		{"cheescake", "cheesecake"},
		{"frnace", "france"},
		{"paris", "paris"},
		{"chillin", "chillin"},
		{"wether", "wether"},
		{"Hello, pasword!", "Hello, password!"},
		{"iPhone", "iPhone"},
		{"2 + 2", "2 + 2"},
	}
	for _, c := range cases {
		if got := b.fix(c.in); got != c.want {
			t.Errorf("%q -> %q, want %q", c.in, got, c.want)
		}
	}
}

func TestHodgeTypos(t *testing.T) {
	path := `C:\Users\grego\models\dev\hodge-5\tokenizer.json`
	if _, err := os.Stat(path); err != nil {
		t.Skip(err)
	}
	got, ok := Adapt(path, "how do I reset my pasword in frnace")
	if !ok {
		t.Fatal(got)
	}
	if !strings.Contains(got, "password") || !strings.Contains(got, "france") {
		t.Fatalf("%q", got)
	}
	same, changed := Adapt(path, "what is the capital of France")
	if changed {
		t.Fatalf("kept sentence changed to %q", same)
	}
	cake, _ := Adapt(path, "cheescake")
	if cake != "cheesecake" {
		t.Fatalf("cheescake -> %q", cake)
	}
}

func TestOSA(t *testing.T) {
	if d := osa("recieve", "receive", 2); d != 1 {
		t.Fatal(d)
	}
	if d := osa("cheescake", "cheesecake", 2); d != 1 {
		t.Fatal(d)
	}
	if d := osa("paris", "pairs", 2); d != 1 {
		t.Fatal(d)
	}
}

// A capital inside a sentence is a name, and all capitals an acronym: neither
// is a typo. The checkpoint's word list had "mon" and not "mona", and "Who
// painted the Mona Lisa?" went out as "the Mon Lisa".
func TestNamesStay(t *testing.T) {
	b := testBook()
	b.rank["mon"] = 50
	id := uint32(len(b.word))
	b.word = append(b.word, "mon")
	for _, d := range deletes("mon") {
		b.dels[d] = append(b.dels[d], id)
	}
	if got := b.fix("Who painted the Mona Lisa?"); got != "Who painted the Mona Lisa?" {
		t.Fatalf("name changed: %q", got)
	}
	if got := b.fix("Is TEH an acronym?"); got != "Is TEH an acronym?" {
		t.Fatalf("acronym changed: %q", got)
	}
	// The first word of a sentence is still read, and a lowercase typo still fixed.
	if got := b.fix("Teh weather. Teh end, teh."); got != "The weather. The end, the." {
		t.Fatalf("typos: %q", got)
	}
}
