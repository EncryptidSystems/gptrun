package spell

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

func load(path string) (*Book, error) {
	vocab, err := vocabOf(path)
	if err != nil {
		return nil, err
	}
	b := &Book{
		rank: map[string]int{},
		hold: map[string]bool{},
		dels: map[string][]uint32{},
	}
	for _, line := range strings.Split(ailza, "\n") {
		w := strings.ToLower(strings.Trim(strings.TrimSpace(line), "'’"))
		if w != "" {
			b.hold[w] = true
		}
	}
	for token, id := range vocab {
		if id < 0 {
			continue
		}
		w, ok := wholeWord(token)
		if !ok {
			continue
		}
		if cur, seen := b.rank[w]; !seen || id < cur {
			b.rank[w] = id
		}
	}
	for w, r := range b.rank {
		if r <= 0 || r >= candRank {
			continue
		}
		if n := utf8.RuneCountInString(w); n < 3 || n > 24 {
			continue
		}
		id := uint32(len(b.word))
		b.word = append(b.word, w)
		for _, d := range deletes(w) {
			b.dels[d] = append(b.dels[d], id)
		}
	}
	return b, nil
}

func wholeWord(token string) (string, bool) {
	s := token
	switch {
	case strings.HasPrefix(s, "Ġ"):
		s = strings.TrimPrefix(s, "Ġ")
	case strings.HasPrefix(s, "▁"):
		s = strings.TrimPrefix(s, "▁")
	}
	if s == "" || strings.Contains(s, " ") {
		return "", false
	}
	s = strings.ToLower(s)
	n := utf8.RuneCountInString(s)
	if n < 3 || n > 24 {
		return "", false
	}
	for _, r := range s {
		if r == '\'' || r == '’' {
			continue
		}
		if !unicode.IsLetter(r) {
			return "", false
		}
	}
	return s, true
}

func vocabOf(path string) (map[string]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	key := []byte(`"vocab":{`)
	i := bytes.Index(data, key)
	if i < 0 {
		return nil, os.ErrNotExist
	}
	dec := json.NewDecoder(bytes.NewReader(data[i+len(key)-1:]))
	var vocab map[string]int
	if err := dec.Decode(&vocab); err != nil {
		return nil, err
	}
	return vocab, nil
}
