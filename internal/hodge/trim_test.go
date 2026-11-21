package hodge

import (
	"errors"
	"os"
	"strings"
	"testing"

	"gptrun/internal/catalog"
)

// TestTrim runs a chat past a small context window. Every turn must succeed;
// the turn after a trim must be the reply a fresh chat of exactly the kept
// turns gives; a message larger than the window is ErrTooLong and the chat
// goes on; a stateless history past the window loses its oldest turns.
// MODELRUN_GOLDEN=model.yaml.
func TestTrim(t *testing.T) {
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
	// The kernels and buffers are sized for the full window; a smaller limit
	// only moves the checks.
	full := m.gpu.ctx
	m.gpu.ctx = 1400
	defer func() { m.gpu.ctx = full }()
	liveMu.Lock()
	live, liveName = m, "trim"
	dropChat()
	liveMu.Unlock()
	defer func() {
		liveMu.Lock()
		live, liveName = nil, ""
		dropChat()
		liveMu.Unlock()
	}()
	var trims int
	logger = func(s string) {
		if strings.HasPrefix(s, "chat trimmed") || strings.HasPrefix(s, "history trimmed") {
			trims++
			t.Log(s)
		}
	}
	defer func() { logger = nil }()

	opt := Opt{N: 40, Temp: 0, TempSet: true}
	filler := strings.Repeat("The river bends past the old mill and the stone bridge. ", 8)
	checked := false
	for i := 0; i < 12; i++ {
		user := filler + "Question " + string(rune('A'+i)) + ": what is in this passage?"
		before := trims
		prior := append([]Turn(nil), chatLog...)
		text, err := Continue("trim", user, opt, nil)
		if err != nil {
			t.Fatalf("turn %d: %v", i+1, err)
		}
		if chat.pos > m.gpu.ctx {
			t.Fatalf("turn %d: chat at %d past the window %d", i+1, chat.pos, m.gpu.ctx)
		}
		if trims > before && !checked {
			// The kept history is the log less this exchange.
			kept := chatLog[:len(chatLog)-2]
			if len(kept) == 0 || len(kept) >= len(prior) {
				t.Fatalf("turn %d: trimmed but kept %d of %d turns", i+1, len(kept), len(prior))
			}
			// The fresh chat asks the same question the trimmed chat asked,
			// including an earlier basin when one was read back in.
			ref, err := m.transcript(append(append([]Turn(nil), kept...), Turn{Role: "user", Content: promptedUser}), opt, nil)
			if err != nil {
				t.Fatal(err)
			}
			if ref != text {
				t.Fatalf("turn %d after the trim:\nchat:  %q\nfresh: %q", i+1, text, ref)
			}
			checked = true
			t.Logf("turn %d: trimmed to %d turns; reply matches a fresh chat of them", i+1, len(kept))
		}
	}
	if !checked {
		t.Fatal("the chat never trimmed")
	}

	// One message larger than the window goes through the holonomy–eikonal
	// field: its head is written there, its tail is the turn, and the chat
	// stays inside the window.
	_, err = Continue("trim", strings.Repeat(filler, 14)+"What does the passage describe?", opt, nil)
	if errors.Is(err, ErrTooLong) || err != nil {
		t.Fatalf("huge message: %v", err)
	}
	if chat.pos > m.gpu.ctx {
		t.Fatalf("huge message: chat at %d past the window %d", chat.pos, m.gpu.ctx)
	}
	if chatHES == nil || chatHES.Writes() == 0 {
		t.Fatal("huge message: nothing was written into the field")
	}
	if _, err := Continue("trim", "Say hello.", opt, nil); err != nil {
		t.Fatalf("after a huge message: %v", err)
	}

	// A stateless history past the window.
	var turns []Turn
	for i := 0; i < 14; i++ {
		turns = append(turns, Turn{Role: "user", Content: filler}, Turn{Role: "assistant", Content: "It is about a river."})
	}
	turns = append(turns, Turn{Role: "user", Content: "What is it about?"})
	before := trims
	if _, err := m.transcript(turns, opt, nil); err != nil {
		t.Fatalf("long stateless history: %v", err)
	}
	if trims == before {
		t.Fatal("stateless history was not trimmed")
	}
}
