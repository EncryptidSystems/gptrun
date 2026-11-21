// Package spell recovers the word a typo was aiming at.
//
// The book is the checkpoint's own whole words, the ones it was trained to
// read as words. Ailza's list is held as written, so those spellings stay.
// A token changes only when one known word is clearly nearer than any other.
// The person's own letters stay on screen. The model hears the recovered word.
package spell

import (
	_ "embed"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

//go:embed ailza.txt
var ailza string

const (
	// keepRank is a word common enough to leave alone.
	keepRank = 60000
	// candRank is the rarest word that may replace a typo.
	candRank = 200000
)

// Book is a spell lexicon.
type Book struct {
	rank map[string]int
	hold map[string]bool
	dels map[string][]uint32
	word []string
}

var (
	mu    sync.Mutex
	books = map[string]*Book{}
	fails = map[string]error{}
)

// Adapt rewrites typos in text using the checkpoint tokenizer at path.
// The bool is true when any word changed. A missing tokenizer leaves the
// text as written.
func Adapt(tokenizer, text string) (string, bool) {
	if strings.TrimSpace(text) == "" || tokenizer == "" {
		return text, false
	}
	b, err := open(tokenizer)
	if err != nil || b == nil {
		return text, false
	}
	out := b.fix(text)
	return out, out != text
}

func open(path string) (*Book, error) {
	mu.Lock()
	defer mu.Unlock()
	if b, ok := books[path]; ok {
		return b, nil
	}
	if err, ok := fails[path]; ok {
		return nil, err
	}
	b, err := load(path)
	if err != nil {
		fails[path] = err
		return nil, err
	}
	books[path] = b
	return b, nil
}

func (b *Book) fix(text string) string {
	var out strings.Builder
	out.Grow(len(text))
	// start is true at the first word of a sentence. A capital there says
	// nothing; anywhere else it marks a name, which stays as written.
	start := true
	i := 0
	for i < len(text) {
		r, size := utf8.DecodeRuneInString(text[i:])
		if !unicode.IsLetter(r) {
			if r == '.' || r == '!' || r == '?' || r == '\n' {
				start = true
			}
			out.WriteRune(r)
			i += size
			continue
		}
		j := i
		for j < len(text) {
			r, size = utf8.DecodeRuneInString(text[j:])
			if unicode.IsLetter(r) || r == '\'' || r == '’' {
				j += size
				continue
			}
			break
		}
		word := text[i:j]
		if first, _ := utf8.DecodeRuneInString(word); !start && unicode.IsUpper(first) {
			out.WriteString(word)
		} else {
			out.WriteString(b.wordOf(word))
		}
		start = false
		i = j
	}
	return out.String()
}

func (b *Book) wordOf(src string) string {
	if !mayChange(src) {
		return src
	}
	key := strings.ToLower(strings.Trim(src, "'’"))
	if key == "" || b.hold[key] {
		return src
	}
	next := b.nearest(key)
	if next == "" || next == key {
		return src
	}
	return applyCase(src, next)
}

func mayChange(src string) bool {
	rs := []rune(src)
	if len(rs) < 3 || len(rs) > 24 {
		return false
	}
	upper := 0
	for _, r := range rs {
		if unicode.IsUpper(r) {
			upper++
		}
	}
	// All capitals is an acronym (NASA, HTML): left as written.
	if upper == 0 {
		return true
	}
	return upper == 1 && unicode.IsUpper(rs[0])
}

func applyCase(src, repl string) string {
	rs := []rune(src)
	upper := 0
	for _, r := range rs {
		if unicode.IsUpper(r) {
			upper++
		}
	}
	if upper == len(rs) {
		return strings.ToUpper(repl)
	}
	if upper == 1 && unicode.IsUpper(rs[0]) {
		rr := []rune(repl)
		if len(rr) == 0 {
			return repl
		}
		rr[0] = unicode.ToUpper(rr[0])
		return string(rr)
	}
	return repl
}

type hit struct {
	w    string
	rank int
	dist int
}

func (b *Book) nearest(word string) string {
	own, known := b.rank[word]
	if known && own > 0 && own <= keepRank {
		return word
	}
	n := utf8.RuneCountInString(word)
	seen := map[string]bool{}
	var hits []hit
	consider := func(cand string) {
		if cand == "" || cand == word || seen[cand] {
			return
		}
		seen[cand] = true
		cr, ok := b.rank[cand]
		if !ok || cr <= 0 || cr >= candRank {
			return
		}
		d := osa(word, cand, 2)
		if d < 1 || d > 2 {
			return
		}
		if known && cr*8 >= own {
			return
		}
		if d == 2 && (known || n < 8 || cr > keepRank) {
			return
		}
		hits = append(hits, hit{w: cand, rank: cr, dist: d})
	}
	for _, id := range b.dels[word] {
		consider(b.word[id])
	}
	for _, d := range deletes(word) {
		if cr, ok := b.rank[d]; ok && cr > 0 && cr < candRank {
			consider(d)
		}
		for _, id := range b.dels[d] {
			consider(b.word[id])
		}
	}
	minD := 3
	for _, h := range hits {
		if h.dist < minD {
			minD = h.dist
		}
	}
	bestR := int(^uint(0) >> 1)
	second := bestR
	bestW := ""
	for _, h := range hits {
		if h.dist != minD {
			continue
		}
		if h.rank < bestR {
			second = bestR
			bestR = h.rank
			bestW = h.w
		} else if h.rank < second {
			second = h.rank
		}
	}
	if bestW == "" || second < bestR*3 {
		return word
	}
	return bestW
}

func deletes(word string) []string {
	rs := []rune(word)
	if len(rs) < 2 {
		return nil
	}
	out := make([]string, 0, len(rs))
	for i := range rs {
		out = append(out, string(rs[:i])+string(rs[i+1:]))
	}
	return out
}

// osa is the optimal string alignment distance, capped at max.
// A distance above max is returned as max+1.
func osa(a, b string, max int) int {
	ar, br := []rune(a), []rune(b)
	la, lb := len(ar), len(br)
	if la < lb {
		ar, br = br, ar
		la, lb = lb, la
	}
	if la-lb > max {
		return max + 1
	}
	prev := make([]int, lb+1)
	cur := make([]int, lb+1)
	back := make([]int, lb+1)
	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		cur[0] = i
		rowMin := cur[0]
		for j := 1; j <= lb; j++ {
			cost := 1
			if ar[i-1] == br[j-1] {
				cost = 0
			}
			del := prev[j] + 1
			ins := cur[j-1] + 1
			sub := prev[j-1] + cost
			d := del
			if ins < d {
				d = ins
			}
			if sub < d {
				d = sub
			}
			if i > 1 && j > 1 && ar[i-1] == br[j-2] && ar[i-2] == br[j-1] {
				if t := back[j-2] + 1; t < d {
					d = t
				}
			}
			cur[j] = d
			if d < rowMin {
				rowMin = d
			}
		}
		if rowMin > max {
			return max + 1
		}
		back, prev, cur = prev, cur, back
	}
	if prev[lb] > max {
		return max + 1
	}
	return prev[lb]
}
