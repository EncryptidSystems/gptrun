package mark

import (
	"math"
	"math/rand"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"gptrun/internal/tokenizers"
)

// vocabCochain is the cochain tensor of vocab.float.safetensors, with the
// letter-word rows kept as centered unit vectors. Similarity is cosine on
// those rows. The key is the one KeyFrom already extracts.
type vocabCochain struct {
	data []float64
	rows int
	cols int
	mu   []float64
	tok  *tokenizers.Tokenizer
	ids  []int
	unit []float32
	word []string
}

func loadVocabCochain(path, tokenizer string) (*vocabCochain, error) {
	data, rows, cols, _, err := LoadMatrix(path, "cochain")
	if err != nil {
		return nil, err
	}
	tok, err := tokenizers.Load(tokenizer)
	if err != nil {
		return nil, err
	}
	v := &vocabCochain{data: data, rows: rows, cols: cols, tok: tok}
	v.mu = make([]float64, cols)
	for id := 0; id < rows; id++ {
		base := id * cols
		for g := 0; g < cols; g++ {
			v.mu[g] += data[base+g]
		}
	}
	for g := range v.mu {
		v.mu[g] /= float64(rows)
	}
	for id := 0; id < rows && id < tok.Size(); id++ {
		if tok.IsSpecial(id) {
			continue
		}
		raw := tok.Decode([]int{id}, false)
		// A word token in this byte-level vocabulary begins with a space.
		// A piece of a word does not, and it is not a similar word.
		if !strings.HasPrefix(raw, " ") {
			continue
		}
		w := strings.TrimSpace(raw)
		if !letterWord(w) {
			continue
		}
		u := make([]float32, cols)
		var n2 float64
		base := id * cols
		for g := 0; g < cols; g++ {
			d := data[base+g] - v.mu[g]
			u[g] = float32(d)
			n2 += d * d
		}
		if n2 == 0 {
			continue
		}
		s := float32(1 / math.Sqrt(n2))
		for g := range u {
			u[g] *= s
		}
		v.ids = append(v.ids, id)
		v.word = append(v.word, w)
		v.unit = append(v.unit, u...)
	}
	return v, nil
}

// tokenIDs is the placement cochainkey uses: every non-special id.
func (v *vocabCochain) tokenIDs(text string) []int {
	var ids []int
	for _, id := range v.tok.Encode(text, true) {
		if v.tok.IsSpecial(id) {
			continue
		}
		ids = append(ids, id)
	}
	return ids
}

// centroid is the corpus placement KeyFrom reads.
func (v *vocabCochain) centroid(text string) []float64 {
	return RowsCentroid(v.data, v.rows, v.cols, v.tokenIDs(text))
}

type neighbor struct {
	Word string
	Cos  float64
	Proj float64
}

// nearest returns the closest other letter words. Proj is the cosine of the
// centered row with key, which is empty when key is nil.
func (v *vocabCochain) nearest(word string, key []float64, k int) []neighbor {
	ids := v.tokenIDs(" " + word)
	if len(ids) == 0 {
		return nil
	}
	q := make([]float64, v.cols)
	for _, id := range ids {
		if id < 0 || id >= v.rows {
			continue
		}
		base := id * v.cols
		for g := 0; g < v.cols; g++ {
			q[g] += v.data[base+g] - v.mu[g]
		}
	}
	var n2 float64
	for _, x := range q {
		n2 += x * x
	}
	if n2 == 0 {
		return nil
	}
	s := 1 / math.Sqrt(n2)
	qf := make([]float32, v.cols)
	for g, x := range q {
		qf[g] = float32(x * s)
	}
	type pair struct {
		i   int
		cos float32
	}
	best := make([]pair, 0, k)
	push := func(p pair) {
		if len(best) < k {
			best = append(best, p)
			return
		}
		min := 0
		for i := 1; i < len(best); i++ {
			if best[i].cos < best[min].cos {
				min = i
			}
		}
		if p.cos > best[min].cos {
			best[min] = p
		}
	}
	for i := range v.ids {
		if strings.EqualFold(v.word[i], word) || !similarWord(word, v.word[i]) {
			continue
		}
		row := v.unit[i*v.cols : (i+1)*v.cols]
		var c float32
		for g := range qf {
			c += qf[g] * row[g]
		}
		push(pair{i, c})
	}
	sort.Slice(best, func(i, j int) bool { return best[i].cos > best[j].cos })
	out := make([]neighbor, len(best))
	for i, p := range best {
		nb := neighbor{Word: v.word[p.i], Cos: float64(p.cos)}
		if key != nil {
			row := v.unit[p.i*v.cols : (p.i+1)*v.cols]
			var d float64
			for g, x := range key {
				d += float64(row[g]) * x
			}
			nb.Proj = d
		}
		out[i] = nb
	}
	return out
}

// similarWord rejects a byte-level piece that is only a short prefix of the
// word it sits inside. mist/misty and arch/arches stay.
func similarWord(from, to string) bool {
	a := []rune(strings.ToLower(from))
	b := []rune(strings.ToLower(to))
	if len(b) < 3 {
		return false
	}
	af, bf := string(a), string(b)
	if strings.HasPrefix(af, bf) && len(b)*2 <= len(a) {
		return false
	}
	if strings.HasPrefix(bf, af) && len(a)*2 <= len(b) {
		return false
	}
	return true
}

func keyDirection(key CorpusKey, cols int) []float64 {
	kv := make([]float64, cols)
	mask := key.Mask
	if len(mask) > cols {
		mask = mask[:cols]
	}
	var n2 float64
	for g := 0; g < len(mask); g++ {
		switch mask[g] {
		case '+':
			kv[g] = 1
		case '-':
			kv[g] = -1
		default:
			continue
		}
		n2 += 1
	}
	if n2 == 0 {
		return nil
	}
	s := 1 / math.Sqrt(n2)
	for g := range kv {
		kv[g] *= s
	}
	return kv
}

func wordProj(v *vocabCochain, word string, key []float64) float64 {
	ids := v.tokenIDs(" " + word)
	q := make([]float64, v.cols)
	for _, id := range ids {
		if id < 0 || id >= v.rows {
			continue
		}
		base := id * v.cols
		for g := 0; g < v.cols; g++ {
			q[g] += v.data[base+g] - v.mu[g]
		}
	}
	var n2, d float64
	for g, x := range q {
		n2 += x * x
		if g < len(key) {
			d += x * key[g]
		}
	}
	if n2 == 0 {
		return 0
	}
	return d / math.Sqrt(n2)
}

// FlipKind chooses which similar word replaces a letter word.
type FlipKind int

const (
	FlipNearest FlipKind = iota
	FlipKey
)

// FlipStep is one replacement.
type FlipStep struct {
	From string
	To   string
	Cos  float64
}

// FlipReport is one rewritten text. NearMean is the mean cosine of each
// distinct letter word to its nearest other vocabulary word.
type FlipReport struct {
	Text     string
	Steps    []FlipStep
	Words    int
	NearMean float64
	NearOK   int
}

// FlipText rewrites letter words as similar vocabulary words. FlipKey uses
// the corpus key: among the eight nearest neighbors it keeps the one whose
// cochain row most opposes the key, and only when that opposition is stronger
// than the original word. A neighbor below minCos is not similar.
func (v *vocabCochain) FlipText(text string, key CorpusKey, kind FlipKind, minCos float64) FlipReport {
	var kv []float64
	if kind == FlipKey {
		kv = keyDirection(key, v.cols)
	}
	cache := map[string]string{}
	var rep FlipReport
	var nearSum float64
	var b strings.Builder
	for i := 0; i < len(text); {
		w, end, ok := letterSpan(text, i)
		if end == i {
			r, size := utf8.DecodeRuneInString(text[i:])
			if r == utf8.RuneError && size == 1 {
				b.WriteByte(text[i])
				i++
				continue
			}
			b.WriteRune(r)
			i += size
			continue
		}
		if !ok {
			b.WriteString(text[i:end])
			i = end
			continue
		}
		low := strings.ToLower(w)
		if to, seen := cache[low]; seen {
			b.WriteString(to)
			i = end
			continue
		}
		rep.Words++
		nbs := v.nearest(w, kv, 8)
		to := w
		var cos float64
		if len(nbs) > 0 {
			nearSum += nbs[0].Cos
			if nbs[0].Cos >= minCos {
				rep.NearOK++
			}
		}
		if kind == FlipNearest {
			if len(nbs) > 0 && nbs[0].Cos >= minCos {
				to = nbs[0].Word
				cos = nbs[0].Cos
			}
		} else if kv != nil {
			orig := wordProj(v, w, kv)
			best := -1
			for i, nb := range nbs {
				if nb.Cos < minCos || nb.Proj >= orig {
					continue
				}
				if best < 0 || nb.Proj < nbs[best].Proj {
					best = i
				}
			}
			if best >= 0 {
				to = nbs[best].Word
				cos = nbs[best].Cos
			}
		}
		cache[low] = to
		if to != w {
			rep.Steps = append(rep.Steps, FlipStep{From: w, To: to, Cos: cos})
		}
		b.WriteString(to)
		i = end
	}
	rep.Text = b.String()
	if rep.Words > 0 {
		rep.NearMean = nearSum / float64(rep.Words)
	}
	return rep
}

func letterSpan(s string, i int) (string, int, bool) {
	j := i
	letters := 0
	for j < len(s) {
		r, size := utf8.DecodeRuneInString(s[j:])
		if unicode.IsLetter(r) {
			j += size
			letters++
			continue
		}
		if r == '\'' && letters > 0 && j+1 < len(s) {
			n, nsize := utf8.DecodeRuneInString(s[j+1:])
			if unicode.IsLetter(n) {
				j += 1 + nsize
				continue
			}
		}
		break
	}
	if j == i {
		return "", i, false
	}
	return s[i:j], j, letterWord(s[i:j])
}

// wordRows is the watermark placement: one raw sum per distinct letter word.
// A repeated word is masked, as Score does.
func (v *vocabCochain) wordRows(text string) [][]float64 {
	seen := map[string]bool{}
	var xs [][]float64
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
		k := strings.ToLower(w)
		if seen[k] {
			continue
		}
		seen[k] = true
		row := make([]float64, v.cols)
		for _, id := range v.tokenIDs(w) {
			if id < 0 || id >= v.rows {
				continue
			}
			base := id * v.cols
			for g := 0; g < v.cols; g++ {
				row[g] += v.data[base+g]
			}
		}
		xs = append(xs, row)
	}
	return xs
}

// marked is the gptrun watermark: cell-gradient share above the vocabulary
// null, on a text of at least 24 distinct letter words.
func (v *vocabCochain) marked(text string) (words int, share, nullMax float64, ok bool) {
	xs := v.wordRows(text)
	words = len(xs)
	if words == 0 {
		return 0, 0, 0, false
	}
	share = cellShare(xs, v.mu)
	nullMax = v.cellNull(words)
	ok = words >= 24 && share > nullMax
	return words, share, nullMax, ok
}

func (v *vocabCochain) cellNull(n int) float64 {
	rng := rand.New(rand.NewSource(int64(n) + 1))
	const trials = 16
	var cellMax float64
	pool := v.ids
	if len(pool) == 0 {
		return 0
	}
	for t := 0; t < trials; t++ {
		rows := make([][]float64, n)
		used := map[int]bool{}
		for i := 0; i < n; i++ {
			id := pool[rng.Intn(len(pool))]
			for used[id] && len(used) < len(pool) {
				id = pool[rng.Intn(len(pool))]
			}
			used[id] = true
			row := make([]float64, v.cols)
			base := id * v.cols
			for g := 0; g < v.cols; g++ {
				row[g] = v.data[base+g]
			}
			rows[i] = row
		}
		sh := cellShare(rows, v.mu)
		if t == 0 || sh > cellMax {
			cellMax = sh
		}
	}
	return cellMax
}

// maskHits is how many agreed cells of key have the same sign on cent.
func maskHits(cent []float64, key CorpusKey) (hit, agreed int) {
	for g := 0; g < len(key.Mask) && g < len(cent); g++ {
		switch key.Mask[g] {
		case '+', '-':
		default:
			continue
		}
		agreed++
		s := byte('+')
		if cent[g] < 0 {
			s = '-'
		}
		if key.Mask[g] == s {
			hit++
		}
	}
	return hit, agreed
}
