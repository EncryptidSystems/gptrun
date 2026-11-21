package hodge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"testing"

	"gptrun/internal/catalog"
)

type goldenStep struct {
	ID   int    `json:"id"`
	Hash uint64 `json:"hash"`
}

type golden struct {
	Prompt []int        `json:"prompt"`
	Steps  []goldenStep `json:"steps"`
}

func hashLogits(lg []float32) uint64 {
	h := fnv.New64a()
	b := make([]byte, 4)
	for _, v := range lg {
		binary.LittleEndian.PutUint32(b, math.Float32bits(v))
		h.Write(b)
	}
	return h.Sum64()
}

// TestGolden records greedy ids and a hash of every step's logits, or compares
// them with a recording. MODELRUN_GOLDEN=model.yaml, MODELRUN_GOLDEN_FILE=out.json,
// MODELRUN_GOLDEN_WRITE=1 records.
func TestGolden(t *testing.T) {
	cfg := os.Getenv("MODELRUN_GOLDEN")
	file := os.Getenv("MODELRUN_GOLDEN_FILE")
	if cfg == "" || file == "" {
		t.Skip("MODELRUN_GOLDEN and MODELRUN_GOLDEN_FILE are not set")
	}
	meta, err := catalog.Load(cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := open(meta)
	if err != nil {
		t.Fatal(err)
	}
	d := &dialog{}
	ids := turnIDs(m.tok, m.kind, m.bos, "You are a helpful assistant.", "Explain how a rainbow forms.", true)
	if err := m.prefill(d, ids); err != nil {
		t.Fatal(err)
	}
	var got golden
	got.Prompt = ids
	var recent []int
	keep := func(id int) bool { return m.stop[id] || m.tok.IsSpecial(id) }
	ends := m.stopIDs()
	kept := map[string][]int{}
	for i := 0; i < 24; i++ {
		// The full head: the recorded logits, bit for bit.
		if err := m.head(d, &headReq{exact: true}); err != nil {
			t.Fatal(err)
		}
		full := append([]float32(nil), d.lg...)
		h := hashLogits(full)
		id := draw(append([]float32(nil), full...), recent, 0, 0, keep, ends, nil)
		got.Steps = append(got.Steps, goldenStep{ID: id, Hash: h})
		// The screened head, greedy and at 0.6: every row it computes is the
		// full head's value, and the draw is the same draw.
		if m.gpu.screen != nil {
			for _, temp := range []float64{0, 0.6} {
				if err := m.head(d, &headReq{temp: temp, recent: recent, keep: keep}); err != nil {
					t.Fatal(err)
				}
				n := 0
				for r, v := range d.lg {
					if math.IsInf(float64(v), -1) {
						continue
					}
					n++
					if math.Float32bits(v) != math.Float32bits(full[r]) {
						t.Fatalf("step %d temp %v row %d: screened %v, full %v", i, temp, r, v, full[r])
					}
				}
				key := fmt.Sprintf("temp %v", temp)
				kept[key] = append(kept[key], n)
				for seed := int64(0); seed < 8; seed++ {
					var r1, r2 *rand.Rand
					if temp > 0 {
						r1, r2 = rand.New(rand.NewSource(seed)), rand.New(rand.NewSource(seed))
					}
					a := draw(append([]float32(nil), full...), recent, temp, 0.9, keep, ends, r1)
					b := draw(append([]float32(nil), d.lg...), recent, temp, 0.9, keep, ends, r2)
					if a != b {
						t.Fatalf("step %d temp %v seed %d: screened draw %d, full draw %d", i, temp, seed, b, a)
					}
					if temp == 0 {
						break
					}
				}
			}
		}
		recent = remember(recent, id, repeatLast)
		if err := m.feed(d, id, nil); err != nil {
			t.Fatal(err)
		}
	}
	for k, v := range kept {
		t.Logf("screen %s: rows rescored per step %v", k, v)
	}
	if os.Getenv("MODELRUN_GOLDEN_WRITE") == "1" {
		b, _ := json.MarshalIndent(got, "", " ")
		if err := os.WriteFile(file, b, 0o644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s: %q", file, m.show(idsOf(got.Steps)))
		return
	}
	b, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}
	var want golden
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	t.Logf("text %q", m.show(idsOf(got.Steps)))
	same := 0
	for i := range want.Steps {
		if i >= len(got.Steps) {
			break
		}
		w, g := want.Steps[i], got.Steps[i]
		if w.ID != g.ID {
			t.Errorf("step %d: id %d, want %d", i, g.ID, w.ID)
			break
		}
		if w.Hash == g.Hash {
			same++
		}
	}
	t.Logf("ids match; logits bit-identical on %d/%d steps", same, len(want.Steps))
}

func idsOf(s []goldenStep) []int {
	out := make([]int, len(s))
	for i, x := range s {
		out[i] = x.ID
	}
	return out
}
