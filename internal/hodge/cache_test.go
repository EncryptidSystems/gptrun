package hodge

import (
	"os"
	"strings"
	"testing"

	"gptrun/internal/catalog"
)

// TestCacheShared: two dialogs share the device's one cache. A dialog whose
// rows another one wrote over gets them back, and a dialog that starts with a
// cached prefix reuses it; either way its logits are the bits a dialog with the
// cache to itself gets. MODELRUN_GOLDEN=model.yaml.
func TestCacheShared(t *testing.T) {
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
	var reused, replayed bool
	logger = func(s string) {
		if strings.HasPrefix(s, "reusing") {
			reused = true
			t.Log(s)
		}
		if strings.HasPrefix(s, "replaying") || strings.HasPrefix(s, "kv shelf  ") && strings.HasSuffix(s, "back") {
			replayed = true
			t.Log(s)
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
	p1 := turnIDs(m.tok, m.kind, m.bos, "You are a helpful assistant.", "Name a color.", true)
	p2 := turnIDs(m.tok, m.kind, m.bos, "You are terse.", "Count to five.", true)

	// The reference: A alone.
	m.gpu.kv = nil
	ref := &dialog{}
	if err := m.prefill(ref, p1); err != nil {
		t.Fatal(err)
	}
	refIDs, refHash := greedy(ref, 12)

	// A for 6 tokens, then B overwrites the cache, then A for 6 more.
	m.gpu.kv = nil
	a := &dialog{}
	if err := m.prefill(a, p1); err != nil {
		t.Fatal(err)
	}
	first, _ := greedy(a, 6)
	b := &dialog{}
	if err := m.prefill(b, p2); err != nil {
		t.Fatal(err)
	}
	greedy(b, 4)
	second, hash := greedy(a, 6)
	got := append(first, second...)
	for i := range refIDs {
		if got[i] != refIDs[i] {
			t.Fatalf("interleaved: token %d is %d, alone %d", i, got[i], refIDs[i])
		}
	}
	if hash != refHash {
		t.Fatalf("interleaved: final logits differ")
	}

	// C starts with A's whole history as its prompt: all but its last token
	// come from the cache.
	c := &dialog{}
	if err := m.prefill(c, append(append([]int(nil), p1...), refIDs...)); err != nil {
		t.Fatal(err)
	}
	if err := m.head(c, exact); err != nil {
		t.Fatal(err)
	}
	if hashLogits(c.lg) != refHash {
		t.Fatalf("reused prefix: logits differ")
	}
	if !reused || !replayed {
		t.Fatalf("reused %v, replayed %v: a path did not run", reused, replayed)
	}
	t.Logf("interleaved and reused dialogs match the reference: %q", m.show(refIDs))
}
