package mark

import (
	"fmt"
	"math/rand"
	"os"
	"sort"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestGScore(t *testing.T) {
	vocab := `C:\Users\grego\models\prod\2b4t\vocab.float.safetensors`
	tok := `C:\Users\grego\models\prod\2b4t\tokenizer.json`
	if _, err := os.Stat(vocab); err != nil {
		t.Skip(err)
	}
	v, err := loadVocabCochain(vocab, tok)
	if err != nil {
		t.Fatal(err)
	}
	names, texts := readPoems(t, `C:\Users\grego\watermarks`)
	chains := make([]Chain, len(names))
	for i, name := range names {
		chains[i] = Chain{Name: name, Centroid: v.centroid(texts[i])}
	}
	key, err := KeyFrom(chains)
	if err != nil {
		t.Fatal(err)
	}
	dir := keyDirection(key, v.cols)
	if dir == nil {
		t.Fatal("empty key direction")
	}

	seen := map[string]bool{}
	var poem []float64
	var words []string
	for _, text := range texts {
		for i := 0; i < len(text); {
			w, end, ok := letterSpan(text, i)
			if end == i {
				_, size := utf8.DecodeRuneInString(text[i:])
				if size < 1 {
					break
				}
				i += size
				continue
			}
			i = end
			if !ok {
				continue
			}
			low := strings.ToLower(w)
			if seen[low] {
				continue
			}
			seen[low] = true
			words = append(words, w)
			poem = append(poem, wordProj(v, w, dir))
		}
	}
	rng := rand.New(rand.NewSource(1))
	const nRand = 4000
	base := make([]float64, 0, nRand)
	used := map[int]bool{}
	for len(base) < nRand && len(used) < len(v.ids) {
		i := rng.Intn(len(v.ids))
		if used[i] {
			continue
		}
		used[i] = true
		if seen[strings.ToLower(v.word[i])] {
			continue
		}
		base = append(base, projUnit(v, i, dir))
	}
	t.Logf("g = cosine of the centered vocab row with the mask direction")
	t.Logf("poem words %d  mean %.4f  median %.4f  positive %.3f",
		len(poem), meanG(poem), medianOf(poem), fracPos(poem))
	t.Logf("other words %d  mean %.4f  median %.4f  positive %.3f",
		len(base), meanG(base), medianOf(base), fracPos(base))

	type pair struct {
		w string
		g float64
	}
	ps := make([]pair, len(words))
	for i := range words {
		ps[i] = pair{words[i], poem[i]}
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].g > ps[j].g })
	var hi, lo []string
	for _, p := range ps[:12] {
		hi = append(hi, fmt.Sprintf("%s %.3f", p.w, p.g))
	}
	for _, p := range ps[len(ps)-12:] {
		lo = append(lo, fmt.Sprintf("%s %.3f", p.w, p.g))
	}
	t.Logf("highest %s", strings.Join(hi, ", "))
	t.Logf("lowest %s", strings.Join(lo, ", "))
}

func medianOf(xs []float64) float64 {
	cp := append([]float64(nil), xs...)
	sort.Float64s(cp)
	return cp[len(cp)/2]
}
