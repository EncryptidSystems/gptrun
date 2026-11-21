package hodge

import (
	"encoding/gob"
	"hash/fnv"
	"math"
	"os"
	"strings"
	"unicode"

	"gptrun/internal/hes"
)

// Keys for the holonomy–eikonal field. The plain key (keyOf) is the mean of
// the tokens' embedding rows, and its cosine ranking caps what any φ can
// recall. A field also knows its own text, so it can weight what is rare in
// it: the token and word document frequencies over the passages written so
// far. keyStats carries those counts and builds the weighted key.

// keyStats is the document frequency of token ids and words over a field's
// passages.
type keyStats struct {
	Docs  int
	TokDF map[int]int
	WrdDF map[string]int
}

func newKeyStats() *keyStats {
	return &keyStats{TokDF: map[int]int{}, WrdDF: map[string]int{}}
}

// fieldStats is each field's key statistics.
var fieldStats = map[*hes.HES]*keyStats{}

func statsOf(h *hes.HES) *keyStats {
	st, ok := fieldStats[h]
	if !ok {
		st = newKeyStats()
		fieldStats[h] = st
	}
	return st
}

// keyFor is text's key in field h: the hybrid of the IDF-weighted embedding
// mean and the hashed words, weighted by what is rare in h.
func (m *Model) keyFor(h *hes.HES, text string, ids []int) []float32 {
	return m.hybridKey(ids, text, statsOf(h))
}

// keyDim is the length of a key from keyFor.
func (m *Model) keyDim() int { return m.D + lexDim }

// writeKeyed counts text into h's statistics, keys it and writes it.
func (m *Model) writeKeyed(h *hes.HES, text string, ids []int) (hes.WriteInfo, error) {
	statsOf(h).add(ids, text)
	return h.Write(hes.Item{IDs: ids, Key: m.keyFor(h, text, ids), Text: text})
}

// libraryFor is a field holding texts: no recency head start (λ = 0, every
// passage equally kept) and salience below the lightest edge a token metric
// can make (800/κ = 400 at weight 1000, κ 2), so a library keeps every
// passage. Every text is counted before any is keyed, and φ is fitted to the
// keys before the writes.
func (m *Model) libraryFor(texts []string) (*hes.HES, error) {
	h, err := hes.NewHES(hes.Options{Dims: hesDims(), Seed: hesSeed, Lambda: 0, Salience: 200})
	if err != nil {
		return nil, err
	}
	st := statsOf(h)
	ids := make([][]int, len(texts))
	for i, t := range texts {
		ids[i] = m.pieces(t)
		st.add(ids[i], t)
	}
	keys := make([][]float32, len(texts))
	var fit [][]float32
	for i, t := range texts {
		keys[i] = m.keyFor(h, t, ids[i])
		if keys[i] != nil {
			fit = append(fit, keys[i])
		}
	}
	if err := h.FitKeys(fit); err != nil {
		return nil, err
	}
	for i, t := range texts {
		if _, err := h.Write(hes.Item{IDs: ids[i], Key: keys[i], Text: t}); err != nil {
			logf("hes write  %v", err)
		}
	}
	return h, nil
}

// newChatField is a field for writes not known in advance, with λ from the
// design rule for chatWindow recent writes.
func newChatField() (*hes.HES, error) {
	return hes.NewHES(hes.Options{Dims: hesDims(), Seed: hesSeed, Lambda: -1, Window: chatWindow})
}

func saveStats(path string, st *keyStats) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := gob.NewEncoder(f).Encode(st); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func loadStats(path string) (*keyStats, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st := newKeyStats()
	if err := gob.NewDecoder(f).Decode(st); err != nil {
		return nil, err
	}
	return st, nil
}

// words is the lower-cased alphanumeric words of s, a trailing plural s cut.
func words(s string) []string {
	var out []string
	for _, w := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") {
			w = w[:len(w)-1]
		}
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}

// add counts one passage.
func (k *keyStats) add(ids []int, text string) {
	k.Docs++
	seen := map[int]bool{}
	for _, id := range ids {
		if !seen[id] {
			seen[id] = true
			k.TokDF[id]++
		}
	}
	ws := map[string]bool{}
	for _, w := range words(text) {
		if !ws[w] {
			ws[w] = true
			k.WrdDF[w]++
		}
	}
}

// idf is log((N+1)/(df+½)): near 0 for a word every passage has.
func idf(df, docs int) float64 {
	return math.Log(float64(docs+1) / (float64(df) + 0.5))
}

// lexDim is the size of the hashed word vector.
const lexDim = 2048

// lexAlpha is the lexical share of a hybrid key (the embedding has 1 − α).
var lexAlpha = 0.4

// embKey is the IDF-weighted mean of the tokens' embedding rows, less the
// vocabulary mean, at unit length.
func (m *Model) embKey(ids []int, k *keyStats) []float32 {
	var use []int
	for _, id := range ids {
		if id >= 0 && id < m.Vocab && (m.tok == nil || !m.tok.IsSpecial(id)) {
			use = append(use, id)
		}
	}
	if len(use) == 0 {
		return nil
	}
	rows, err := m.embRows(use)
	if err != nil {
		logf("hes key  %v", err)
		return nil
	}
	acc := make([]float64, m.D)
	var wsum float64
	for _, id := range use {
		r, ok := rows[id]
		if !ok {
			continue
		}
		w := 1.0
		if k != nil && k.Docs > 0 {
			w = idf(k.TokDF[id], k.Docs)
		}
		for j, v := range r {
			acc[j] += w * float64(v)
		}
		wsum += w
	}
	if wsum == 0 {
		return nil
	}
	mean := m.vocabMean()
	out := make([]float32, m.D)
	for j := range acc {
		out[j] = float32(acc[j]/wsum - float64(mean[j]))
	}
	return unitF(out)
}

// lexKey is the hashed, IDF-weighted word counts of text at unit length.
func lexKey(text string, k *keyStats) []float32 {
	v := make([]float32, lexDim)
	any := false
	for _, w := range words(text) {
		h := fnv.New32a()
		h.Write([]byte(w))
		x := h.Sum32()
		wt := 1.0
		if k != nil && k.Docs > 0 {
			df := k.WrdDF[w]
			if df == 0 {
				continue // no passage has it: it can only collide in the hash
			}
			wt = idf(df, k.Docs)
		}
		sign := float32(1)
		if x&1 == 1 {
			sign = -1
		}
		v[(x>>1)%lexDim] += sign * float32(wt)
		any = true
	}
	if !any {
		return nil
	}
	return unitF(v)
}

// hybridKey joins the weighted embedding key and the lexical key, each at
// unit length, as √(1−α)·e ⊕ √α·l: its cosine is (1−α)·cos_e + α·cos_l.
func (m *Model) hybridKey(ids []int, text string, k *keyStats) []float32 {
	e := m.embKey(ids, k)
	l := lexKey(text, k)
	if e == nil && l == nil {
		return nil
	}
	if e == nil {
		e = make([]float32, m.D)
	}
	if l == nil {
		l = make([]float32, lexDim)
	}
	a := float32(math.Sqrt(1 - lexAlpha))
	b := float32(math.Sqrt(lexAlpha))
	out := make([]float32, 0, len(e)+len(l))
	for _, x := range e {
		out = append(out, a*x)
	}
	for _, x := range l {
		out = append(out, b*x)
	}
	return unitF(out)
}

func unitF(v []float32) []float32 {
	var ss float64
	for _, x := range v {
		ss += float64(x) * float64(x)
	}
	if ss == 0 {
		return nil
	}
	inv := float32(1 / math.Sqrt(ss))
	for i := range v {
		v[i] *= inv
	}
	return v
}
