package hodge

import (
	"os"
	"testing"

	"gptrun/internal/agent"
	"gptrun/internal/catalog"
)

// TestSampledExact: a sampled reply at the config temperature is the same
// text on the TUI's fast path (warm-up, a route pass, shelf swaps, the
// screened head) as with the full head and the cache to itself, for the same
// seed. A wrong row anywhere shows up as a different token.
// MODELRUN_GOLDEN=model.yaml.
func TestSampledExact(t *testing.T) {
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
	if m.gpu.screen == nil || m.gpu.shelf == nil {
		t.Fatal("the screened head and the shelf must both be on")
	}
	q := "Tell me about the history and culture of Columbia, Maryland"
	fact := []Turn{
		{Role: "system", Content: agent.FactSystem},
		{Role: "user", Content: "The lookup is already done. Answer the question the person asked, in plain words. Use the notes. " +
			"Give the answer that question asks for. Do not answer with a label, do not use a heading, and do not use JSON.\n" +
			"Notes:\nColumbia, Maryland: Columbia is a census-designated place in Howard County, Maryland. " +
			"It is a planned community of ten villages.\nQuestion: " + q},
	}
	route := []Turn{{Role: "system", Content: agent.RecipeSystem}, {Role: "user", Content: q}}
	opt := Opt{N: 200, ConfigSample: true}
	defer func() { testSeed, testExact = 0, false }()
	shelf := m.gpu.shelf
	for _, seed := range []int64{1, 2, 3} {
		// The TUI's path.
		m.gpu.kv, m.gpu.shelf, shelf.cur = nil, shelf, nil
		testSeed, testExact = 0, false
		if err := m.warm([]string{agent.RecipeSystem, systemFor(m)}); err != nil {
			t.Fatal(err)
		}
		if _, err := m.transcript(route, Opt{N: 16, Temp: 0, TempSet: true}, nil); err != nil {
			t.Fatal(err)
		}
		testSeed = seed
		fast, err := m.transcript(fact, opt, nil)
		if err != nil {
			t.Fatal(err)
		}
		// Full head, nothing shared.
		m.gpu.kv, m.gpu.shelf = nil, nil
		testExact = true
		exact, err := m.transcript(fact, opt, nil)
		if err != nil {
			t.Fatal(err)
		}
		if fast != exact {
			t.Fatalf("seed %d: the fast path's reply differs\nfast:  %q\nexact: %q", seed, fast, exact)
		}
		t.Logf("seed %d: %d bytes identical: %.160q", seed, len(fast), fast)
	}
	m.gpu.shelf = shelf
}
