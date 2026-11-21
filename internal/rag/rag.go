// Package rag pulls Hodge-5 library notes into a prompt. The store lives in
// EmbedDB. A missing store leaves the prompt unchanged.
package rag

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"embeddb"
)

var (
	once    sync.Once
	live    bool
	store   *embeddb.Store
	corpora []*embeddb.Corpus
	weight  *embeddb.Weight
)

// Augment prepends the closest library passages to user. Slash commands and
// an empty library come back unchanged.
//
// Notes stay off unless EMBEDDB_NOTES is 1. Pasting the security library in
// front of every question made the model answer the passage and invent the rest.
func Augment(user string) string {
	user = strings.TrimSpace(user)
	if user == "" || strings.HasPrefix(user, "/") || !notesOn() {
		return user
	}
	once.Do(load)
	if !live {
		return user
	}
	hits := lookup(user)
	if len(hits) == 0 {
		return user
	}
	var b strings.Builder
	b.WriteString("Library notes. Use them when they answer the question.\n\n")
	for i, h := range hits {
		text := h.Text
		if len(text) > 900 {
			text = text[:900]
		}
		fmt.Fprintf(&b, "[%d] %s\n%s\n\n", i+1, filepath.Base(h.Source), strings.TrimSpace(text))
	}
	b.WriteString(user)
	return b.String()
}

func lookup(user string) []embeddb.Hit {
	var hits []embeddb.Hit
	if len(corpora) > 0 {
		found, err := embeddb.SearchLibraries(corpora, weight, user, 12, 8)
		if err == nil {
			hits = append(hits, embeddb.Relevant(user, found)...)
		}
	}
	if len(hits) > 3 {
		hits = hits[:3]
	}
	// The doc store is the project notes. It fills gaps only. It does not
	// get a free slot on every question, which was appending README.md to
	// security answers.
	if store != nil && weight != nil && len(hits) < 3 {
		found, err := embeddb.Query(store, weight, user, 4)
		if err == nil {
			for _, h := range embeddb.Relevant(user, found) {
				hits = append(hits, h)
				if len(hits) == 3 {
					break
				}
			}
		}
	}
	return hits
}

func notesOn() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("EMBEDDB_NOTES"))) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}

func load() {
	corpora = embeddb.OpenCorpora(embeddb.Corpora())
	if len(corpora) > 0 {
		live = true
	}
	path := embeddb.DefaultStore()
	if s, err := embeddb.Open(path); err == nil && s.Len() > 0 {
		store = s
		live = true
	}
	if !live {
		return
	}
	model := embeddb.DefaultModel()
	if store != nil && store.Model != "" {
		model = store.Model
	}
	w, err := embeddb.OpenWeight(model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "embeddb: %v\n", err)
		return
	}
	weight = w
}
