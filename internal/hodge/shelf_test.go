package hodge

import (
	"os"
	"strings"
	"testing"

	"gptrun/internal/catalog"
)

// TestShelf runs the TUI's pattern: a route pass, the chat, a route pass, the
// chat. Every dialog's tokens and final logits must be the bits it gets with
// the cache to itself, with no shelf (replay), the full shelf, and a shelf of
// 8 rows (the rest replayed). MODELRUN_GOLDEN=model.yaml.
func TestShelf(t *testing.T) {
	cfg := os.Getenv("MODELRUN_GOLDEN")
	if cfg == "" {
		t.Skip("MODELRUN_GOLDEN is not set")
	}
	meta, err := catalog.Load(cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := open(meta)
	if err != nil {
		t.Fatal(err)
	}
	g := m.gpu
	full := g.shelf
	if full == nil {
		t.Fatal("no shelf")
	}
	var back, replay int
	logger = func(s string) {
		if strings.HasPrefix(s, "kv shelf  ") && strings.HasSuffix(s, "back") {
			back++
		}
		if strings.HasPrefix(s, "replaying") {
			replay++
		}
	}
	defer func() { logger = nil }()
	exact := &headReq{exact: true}
	greedy := func(d *dialog, n int) ([]int, uint64) {
		var out []int
		for i := 0; i < n; i++ {
			if err := m.head(d, exact); err != nil {
				t.Fatal(err)
			}
			id := choose(d.lg, 0, 0, minP, nil, nil)
			out = append(out, id)
			if err := m.feed(d, id, nil); err != nil {
				t.Fatal(err)
			}
		}
		if err := m.head(d, exact); err != nil {
			t.Fatal(err)
		}
		return out, hashLogits(d.lg)
	}
	chatP := turnIDs(m.tok, m.kind, m.bos, systemText("Hodge"), "Name a color.", true)
	route1 := turnIDs(m.tok, m.kind, m.bos, "Reply with one word.", "Name a color.", true)
	route2 := turnIDs(m.tok, m.kind, m.bos, "Reply with one word.", "Count to five.", true)
	chatMore := turnIDs(m.tok, m.kind, nil, "", "And another?", false)

	alone := func(p []int, n int) ([]int, uint64) {
		g.kv = nil
		g.shelf = nil
		d := &dialog{}
		if err := m.prefill(d, p); err != nil {
			t.Fatal(err)
		}
		return greedy(d, n)
	}
	// References. The chat's second turn follows its own first reply.
	c1, _ := alone(chatP, 8)
	chat2 := append(append(append([]int(nil), chatP...), c1...), chatMore...)
	c2, c2h := alone(chat2, 8)
	r1, r1h := alone(route1, 4)
	r2, r2h := alone(route2, 4)

	same := func(mode, what string, got, want []int, gh, wh uint64) {
		t.Helper()
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: %s token %d is %d, alone %d", mode, what, i, got[i], want[i])
			}
		}
		if gh != wh {
			t.Fatalf("%s: %s final logits differ", mode, what)
		}
	}
	for _, mode := range []string{"replay", "shelf", "shelf of 8", "warm"} {
		g.kv = nil
		g.shelf = nil
		if mode != "replay" {
			g.shelf = full
			full.cur = nil
			full.rows = shelfRows
			if mode == "shelf of 8" {
				full.rows = 8
			}
		}
		if mode == "warm" {
			if err := m.warm([]string{"Reply with one word.", systemText("Hodge")}); err != nil {
				t.Fatal(err)
			}
		}
		back, replay = 0, 0
		ra := &dialog{}
		if err := m.prefill(ra, route1); err != nil {
			t.Fatal(err)
		}
		got, h := greedy(ra, 4)
		same(mode, "route 1", got, r1, h, r1h)

		chat := &dialog{}
		if err := m.prefill(chat, chatP); err != nil {
			t.Fatal(err)
		}
		got, _ = greedy(chat, 8)
		same(mode, "chat 1", got, c1, 0, 0)

		rb := &dialog{}
		if err := m.prefill(rb, route2); err != nil {
			t.Fatal(err)
		}
		got, h = greedy(rb, 4)
		same(mode, "route 2", got, r2, h, r2h)

		if err := m.prefill(chat, chatMore); err != nil {
			t.Fatal(err)
		}
		got, h = greedy(chat, 8)
		same(mode, "chat 2", got, c2, h, c2h)
		t.Logf("%-10s  shelf back %d  replays %d", mode, back, replay)
		switch mode {
		case "replay":
			if back != 0 || replay == 0 {
				t.Fatalf("replay mode: back %d replay %d", back, replay)
			}
		case "shelf":
			if back < 2 || replay != 0 {
				t.Fatalf("shelf mode: back %d replay %d", back, replay)
			}
		case "warm":
			// Every pass, the first route pass and the first chat turn too,
			// starts from rows already read.
			if back < 4 || replay != 0 {
				t.Fatalf("warm: back %d replay %d", back, replay)
			}
		case "shelf of 8":
			if back < 2 || replay == 0 {
				t.Fatalf("small shelf: back %d replay %d", back, replay)
			}
		}
	}
	full.rows = shelfRows
}
