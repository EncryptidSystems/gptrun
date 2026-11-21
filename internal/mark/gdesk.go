package mark

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

//go:embed gdesk.html
var gdeskHTML []byte

// GDesk is the vocabulary g-score of one corpus on one vocab cochain.
type GDesk struct {
	v    *vocabCochain
	key  CorpusKey
	dir  []float64
	snap GSnap
}

// GWord is one vocabulary word and its g-score.
type GWord struct {
	W string  `json:"w"`
	G float64 `json:"g"`
}

// GPoem is one corpus text.
type GPoem struct {
	Name   string  `json:"name"`
	Words  int     `json:"words"`
	Share  float64 `json:"share"`
	Null   float64 `json:"null"`
	Marked bool    `json:"marked"`
	Hits   int     `json:"hits"`
	Agreed int     `json:"agreed"`
}

// GSnap is the page payload.
type GSnap struct {
	Key      string    `json:"key"`
	Code     string    `json:"code"`
	Mask     string    `json:"mask"`
	Agreed   int       `json:"agreed"`
	Cells    int       `json:"cells"`
	Cell63   int       `json:"cell63"`
	PoemN    int       `json:"poemN"`
	BaseN    int       `json:"baseN"`
	PoemMean float64   `json:"poemMean"`
	PoemPos  float64   `json:"poemPos"`
	BaseMean float64   `json:"baseMean"`
	BasePos  float64   `json:"basePos"`
	Words    []GWord   `json:"words"`
	Bins     []float64 `json:"bins"`
	PoemHist []int     `json:"poemHist"`
	BaseHist []int     `json:"baseHist"`
	Poems    []GPoem   `json:"poems"`
}

// OpenGDesk loads the vocab cochain and scores every letter word in the corpus.
// The corpus directory holds one .txt file per text. The key is KeyFrom.
func OpenGDesk(vocab, tokenizer, corpus string) (*GDesk, error) {
	v, err := loadVocabCochain(vocab, tokenizer)
	if err != nil {
		return nil, err
	}
	names, texts, err := readTxtDir(corpus)
	if err != nil {
		return nil, err
	}
	if len(names) < 2 {
		return nil, fmt.Errorf("%s has %d texts", corpus, len(names))
	}
	chains := make([]Chain, len(names))
	for i, name := range names {
		chains[i] = Chain{Name: name, Centroid: v.centroid(texts[i])}
	}
	key, err := KeyFrom(chains)
	if err != nil {
		return nil, err
	}
	dir := keyDirection(key, v.cols)
	if dir == nil {
		return nil, fmt.Errorf("the mask has no agreed cells")
	}
	d := &GDesk{v: v, key: key, dir: dir}
	d.snap = d.build(names, texts)
	return d, nil
}

// Snap is the payload the page draws.
func (d *GDesk) Snap() GSnap { return d.snap }

// ScoreWord is g for one word. ok is false when the word is not a letter word.
func (d *GDesk) ScoreWord(word string) (float64, bool) {
	word = strings.TrimSpace(word)
	if !letterWord(word) {
		return 0, false
	}
	return wordProj(d.v, word, d.dir), true
}

// Serve listens on addr and serves the page. addr is a host:port on loopback.
func (d *GDesk) Serve(addr string) error {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return err
	}
	if host != "127.0.0.1" && host != "localhost" && host != "::1" {
		return fmt.Errorf("gdesk listens on loopback, not %s", host)
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(gdeskHTML)
	})
	mux.HandleFunc("/api/g", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(d.snap)
	})
	mux.HandleFunc("/api/word", func(w http.ResponseWriter, r *http.Request) {
		g, ok := d.ScoreWord(r.URL.Query().Get("w"))
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{"w": strings.TrimSpace(r.URL.Query().Get("w")), "g": g, "ok": ok})
	})
	fmt.Printf("gdesk  http://%s\n", ln.Addr().String())
	return http.Serve(ln, mux)
}

func (d *GDesk) build(names, texts []string) GSnap {
	seen := map[string]bool{}
	var words []GWord
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
			words = append(words, GWord{W: w, G: wordProj(d.v, w, d.dir)})
		}
	}
	sort.Slice(words, func(i, j int) bool { return words[i].G > words[j].G })
	poemG := make([]float64, len(words))
	for i := range words {
		poemG[i] = words[i].G
	}
	rng := rand.New(rand.NewSource(1))
	const nRand = 4000
	base := make([]float64, 0, nRand)
	used := map[int]bool{}
	for len(base) < nRand && len(used) < len(d.v.ids) {
		i := rng.Intn(len(d.v.ids))
		if used[i] {
			continue
		}
		used[i] = true
		if seen[strings.ToLower(d.v.word[i])] {
			continue
		}
		base = append(base, projUnit(d.v, i, d.dir))
	}
	bins := make([]float64, 0, 28)
	for x := -0.45; x < 0.80; x += 0.05 {
		bins = append(bins, round3(x))
	}
	poems := make([]GPoem, len(names))
	for i, name := range names {
		nw, share, null, marked := d.v.marked(texts[i])
		hit, agreed := maskHits(d.v.centroid(texts[i]), d.key)
		poems[i] = GPoem{Name: name, Words: nw, Share: share, Null: null, Marked: marked, Hits: hit, Agreed: agreed}
	}
	cell63 := d.key.Cell63
	return GSnap{
		Key: fmt.Sprintf("%d", d.key.Key), Code: d.key.Code, Mask: d.key.Mask,
		Agreed: d.key.Agreed, Cells: d.key.Cells, Cell63: cell63,
		PoemN: len(poemG), BaseN: len(base),
		PoemMean: meanG(poemG), PoemPos: fracPos(poemG),
		BaseMean: meanG(base), BasePos: fracPos(base),
		Words: words, Bins: bins,
		PoemHist: hist(poemG, bins), BaseHist: hist(base, bins),
		Poems: poems,
	}
}

func hist(xs, bins []float64) []int {
	h := make([]int, len(bins))
	if len(bins) == 0 {
		return h
	}
	width := 0.05
	if len(bins) > 1 {
		width = bins[1] - bins[0]
	}
	for _, x := range xs {
		i := int((x - bins[0]) / width)
		if i < 0 {
			i = 0
		}
		if i >= len(h) {
			i = len(h) - 1
		}
		h[i]++
	}
	return h
}

func round3(x float64) float64 {
	return math.Round(x*1000) / 1000
}

func projUnit(v *vocabCochain, i int, dir []float64) float64 {
	row := v.unit[i*v.cols : (i+1)*v.cols]
	var d float64
	for g, x := range dir {
		d += float64(row[g]) * x
	}
	return d
}

func meanG(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func fracPos(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	n := 0
	for _, x := range xs {
		if x > 0 {
			n++
		}
	}
	return float64(n) / float64(len(xs))
}

func readTxtDir(dir string) ([]string, []string, error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	var names []string
	for _, e := range ents {
		if e.IsDir() || !strings.EqualFold(filepath.Ext(e.Name()), ".txt") {
			continue
		}
		names = append(names, e.Name())
	}
	sort.Strings(names)
	texts := make([]string, len(names))
	for i, name := range names {
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			return nil, nil, err
		}
		texts[i] = string(b)
	}
	return names, texts, nil
}
