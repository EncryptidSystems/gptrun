package mark

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func TestSimilarFlip(t *testing.T) {
	vocab := `C:\Users\grego\models\prod\2b4t\vocab.float.safetensors`
	tok := `C:\Users\grego\models\prod\2b4t\tokenizer.json`
	dir := `C:\Users\grego\watermarks`
	if _, err := os.Stat(vocab); err != nil {
		t.Skip(err)
	}
	v, err := loadVocabCochain(vocab, tok)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("vocab cochain %d x %d  letter words %d", v.rows, v.cols, len(v.ids))
	if len(v.ids) < 1000 {
		t.Fatalf("letter words %d", len(v.ids))
	}
	names, texts := readPoems(t, dir)
	chains := make([]Chain, len(names))
	for i, name := range names {
		chains[i] = Chain{Name: name, Centroid: v.centroid(texts[i])}
	}
	key, err := KeyFrom(chains)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("key %d", key.Key)
	t.Logf("code %s", key.Code)
	t.Logf("mask %s", key.Mask)
	t.Logf("agreed %d of %d  cell63 %d", key.Agreed, key.Cells, key.Cell63)

	const minCos = 0.35
	for hold, name := range names {
		others := make([]Chain, 0, len(chains)-1)
		for i := range chains {
			if i != hold {
				others = append(others, chains[i])
			}
		}
		held, err := KeyFrom(others)
		if err != nil {
			t.Fatal(err)
		}
		text := texts[hold]
		words, share, nullMax, marked := v.marked(text)
		hit, agreed := maskHits(v.centroid(text), held)
		t.Logf("before %s  words %d  marked %v  cell %.4f  null %.4f  key %d/%d  held-out agreed %d",
			name, words, marked, share, nullMax, hit, agreed, held.Agreed)

		near := v.FlipText(text, held, FlipNearest, minCos)
		gWords, gShare, gNull, gMarked := v.marked(near.Text)
		gHit, gAgreed := maskHits(v.centroid(near.Text), held)
		t.Logf("nearest %s  flipped %d/%d  near-mean %.3f  near>=%.2f %d  marked %v  cell %.4f  null %.4f  key %d/%d",
			name, len(near.Steps), near.Words, near.NearMean, minCos, near.NearOK, gMarked, gShare, gNull, gHit, gAgreed)
		t.Logf("nearest sample %s", sampleFlips(near.Steps))

		guided := v.FlipText(text, held, FlipKey, minCos)
		kWords, kShare, kNull, kMarked := v.marked(guided.Text)
		kHit, kAgreed := maskHits(v.centroid(guided.Text), held)
		t.Logf("keyed %s  flipped %d/%d  marked %v  cell %.4f  null %.4f  key %d/%d",
			name, len(guided.Steps), guided.Words, kMarked, kShare, kNull, kHit, kAgreed)
		t.Logf("keyed sample %s", sampleFlips(guided.Steps))
		if kWords == 0 || gWords == 0 {
			t.Errorf("%s rewrite dropped every word", name)
		}
	}
}

func readPoems(t *testing.T, dir string) ([]string, []string) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".txt") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	if len(names) < 2 {
		t.Fatalf("%s has %d texts", dir, len(names))
	}
	texts := make([]string, len(names))
	for i, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		texts[i] = string(b)
	}
	return names, texts
}

func sampleFlips(steps []FlipStep) string {
	n := len(steps)
	if n > 8 {
		n = 8
	}
	parts := make([]string, n)
	for i := 0; i < n; i++ {
		parts[i] = fmt.Sprintf("%s→%s (%.2f)", steps[i].From, steps[i].To, steps[i].Cos)
	}
	if len(parts) == 0 {
		return "none"
	}
	return strings.Join(parts, ", ")
}
