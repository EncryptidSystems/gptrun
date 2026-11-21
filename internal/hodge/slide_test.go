package hodge

import (
	"os"
	"strings"
	"testing"

	"gptrun/internal/catalog"
)

// TestSlide runs one reply longer than a small window. When the reply fills
// the window it slides onto a fresh one through the holonomy–eikonal field and
// keeps going: no "context full", and more text than one window holds.
// MODELRUN_GOLDEN=model.yaml.
func TestSlide(t *testing.T) {
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
	full := m.gpu.ctx
	m.gpu.ctx = 800
	defer func() { m.gpu.ctx = full }()
	var slides int
	logger = func(s string) {
		if strings.HasPrefix(s, "hes slide") {
			slides++
			t.Log(s)
		}
	}
	defer func() { logger = nil }()
	turns := []Turn{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Count from 1 to 600, one number per line, with no other text."},
	}
	text, err := m.transcript(turns, Opt{N: 1000, Temp: 0, TempSet: true}, nil)
	if err != nil {
		t.Fatalf("long reply: %v", err)
	}
	if slides == 0 {
		t.Fatalf("the reply never slid (%d tokens)", len(m.pieces(text)))
	}
	if n := len(m.pieces(text)); n <= m.gpu.ctx {
		t.Fatalf("%d tokens after %d slides: no more than one window", n, slides)
	}
	t.Logf("%d slides, %d tokens, ends %q", slides, len(m.pieces(text)), text[max(0, len(text)-60):])
}
