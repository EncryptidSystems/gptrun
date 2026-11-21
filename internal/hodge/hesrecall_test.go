package hodge

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"gptrun/internal/catalog"
	"gptrun/internal/hes"
)

var recallFacts = [][2]string{
	{"The ferry to Brindle Island leaves at 7:40 every morning from pier 3.", "When does the boat to Brindle Island depart?"},
	{"Bakeries in the old town close at 2 pm on Sundays.", "What time do the old town bakeries shut on Sunday?"},
	{"The museum's dinosaur hall reopened in March after repairs to the roof.", "When did the dinosaur exhibit open again?"},
	{"Parking near the harbour costs 4 dollars an hour after 9 am.", "How much is it to park by the harbour?"},
	{"The library lends telescopes for one week at a time.", "How long can I borrow a telescope from the library?"},
	{"Marta's dog is a grey whippet called Pixel.", "What breed is Marta's dog?"},
	{"The office Wi-Fi password is printed on the fridge in the kitchen.", "Where can I find the Wi-Fi password at work?"},
	{"Our team's quarterly review is on the 14th of December in room 5B.", "Which room is the quarterly review in?"},
	{"The recipe needs 300 grams of flour and two eggs for the pancake batter.", "How much flour goes into the pancakes?"},
	{"The river trail is closed for bridge repairs until the end of May.", "Why is the river path shut?"},
	{"Grandpa's birthday dinner is at the Lotus Garden restaurant on Friday.", "Where are we eating for grandpa's birthday?"},
	{"The printer on the second floor jams if you use thick card.", "Which printer has trouble with card stock?"},
	{"Swimming lessons for children start at 4 pm on Tuesdays and Thursdays.", "What days are the kids' swim classes?"},
	{"The car's spare tyre is under the floor panel in the boot.", "Where is the spare wheel kept in the car?"},
	{"The electricity bill is paid automatically on the first of each month.", "When does the power bill get paid?"},
	{"Sam is allergic to peanuts and carries an epinephrine pen.", "What food allergy does Sam have?"},
	{"The garden shed key hangs on a hook behind the back door.", "Where is the key to the shed?"},
	{"The train from Leeds to York takes about 25 minutes.", "How long is the rail journey from Leeds to York?"},
	{"Our flat's boiler was serviced last October by Hartley Heating.", "Who serviced the boiler?"},
	{"The book club is reading a novel about lighthouse keepers this month.", "What is the book group reading now?"},
	{"Recycling is collected on Wednesday mornings in green bins.", "Which day is recycling picked up?"},
	{"The cat's vaccination is due again in August.", "When does the cat need its next jab?"},
	{"The hiking group meets at the north car park at 8 am on Saturdays.", "Where does the walking club gather?"},
	{"The router is reset by holding the small button for ten seconds.", "How do I reset the internet box?"},
	{"Aunt Rosa lives in Porto and grows lemons on her balcony.", "What fruit does Aunt Rosa grow?"},
	{"The project deadline moved from June to the second week of July.", "When is the project due now?"},
	{"The dentist appointment is on Monday at 10:15 with Dr. Okafor.", "Who is the dentist I am seeing?"},
	{"The coffee machine descaler is in the cupboard under the sink.", "Where is the descaling liquid for the coffee maker?"},
	{"The violin needs new strings before the concert in April.", "What does the violin need before the concert?"},
	{"Tickets for the jazz night cost 18 pounds at the door.", "How much are jazz night tickets?"},
}

// TestLongInputKeepsChunks writes a long text into a library field the way a
// prompt longer than the window is, and checks every chunk still owns ground:
// a library forgets nothing. HES_LONG_TEXT names the text file.
// MODELRUN_GOLDEN=model.yaml.
func TestLongInputKeepsChunks(t *testing.T) {
	cfg, path := os.Getenv("MODELRUN_GOLDEN"), os.Getenv("HES_LONG_TEXT")
	if cfg == "" || path == "" {
		t.Skip("MODELRUN_GOLDEN or HES_LONG_TEXT is not set")
	}
	meta, err := catalog.Load(cfg)
	if err != nil {
		t.Fatal(err)
	}
	m, err := open(meta)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	parts := m.chunks(string(b))
	h, err := m.libraryFor(parts)
	if err != nil {
		t.Fatal(err)
	}
	r := h.Laws()
	t.Logf("%d chunks, %d payloads, %d live, %d landmarks", len(parts), r.Payloads, r.Live, len(h.K.LM))
	if r.Live != r.Payloads {
		for p, n := range h.F.Owners() {
			if n == 0 {
				t.Logf("evicted: %q", snippet(p))
			}
		}
		t.Fatalf("%d of %d chunks lost all their territory", r.Payloads-r.Live, r.Payloads)
	}
}

// TestKeyVariants scores key constructions on the same facts and questions:
// by cosine, and by the HES read on the default torus.
// MODELRUN_GOLDEN=model.yaml.
func TestKeyVariants(t *testing.T) {
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
	st := newKeyStats()
	for _, f := range recallFacts {
		st.add(m.pieces(f[0]), f[0])
	}
	type variant struct {
		name string
		key  func(text string) []float32
	}
	vs := []variant{
		{"mean", func(s string) []float32 { return m.keyOf(m.pieces(s)) }},
		{"idf mean", func(s string) []float32 { return m.embKey(m.pieces(s), st) }},
		{"lexical", func(s string) []float32 { return lexKey(s, st) }},
	}
	for _, a := range []float64{0.3, 0.5, 0.7} {
		a := a
		vs = append(vs, variant{fmt.Sprintf("hybrid α=%.1f", a), func(s string) []float32 {
			old := lexAlpha
			lexAlpha = a
			defer func() { lexAlpha = old }()
			return m.hybridKey(m.pieces(s), s, st)
		}})
	}
	for _, v := range vs {
		var keys, qk [][]float32
		for _, f := range recallFacts {
			keys = append(keys, v.key(f[0]))
			qk = append(qk, v.key(f[1]))
		}
		c1, c3 := 0, 0
		var miss []int
		for i, q := range qk {
			best := make([]int, len(keys))
			sc := make([]float64, len(keys))
			for j, k := range keys {
				best[j] = j
				for x := range k {
					sc[j] += float64(k[x]) * float64(q[x])
				}
			}
			sort.Slice(best, func(a, b int) bool { return sc[best[a]] > sc[best[b]] })
			if best[0] == i {
				c1++
			} else {
				miss = append(miss, i)
				if strings.HasPrefix(v.name, "hybrid α=0.3") {
					t.Logf("  miss %d %q -> %q (%.3f vs own %.3f)", i, recallFacts[i][1], snippet(recallFacts[best[0]][0]), sc[best[0]], sc[i])
				}
			}
			for _, j := range best[:3] {
				if j == i {
					c3++
				}
			}
		}
		h, err := hes.NewHES(hes.Options{Dims: []int{64, 64}, Seed: hesSeed, Lambda: 0, Salience: 200})
		if err != nil {
			t.Fatal(err)
		}
		h.FitKeys(keys)
		for i, f := range recallFacts {
			h.Write(hes.Item{IDs: m.pieces(f[0]), Key: keys[i], Text: f[0]})
		}
		h1, h3 := 0, 0
		for i, q := range qk {
			got := h.ReadAll(q, nil, nil, 3)
			if len(got) > 0 && got[0].Payload == recallFacts[i][0] {
				h1++
			}
			for _, r := range got {
				if r.Payload == recallFacts[i][0] {
					h3++
				}
			}
		}
		t.Logf("%-14s cosine top-1 %2d top-3 %2d   HES top-1 %2d top-3 %2d   cosine misses %v", v.name, c1, c3, h1, h3, miss)
	}
}

// TestRecallByDim measures the full HES read on the model's own keys: for
// each question, is the fact it asks about the first basin read (and in the
// first three), on tori of the same N in dimension 2, 3 and 4? The cosine
// ranking of the same keys is the ceiling φ can at best preserve.
// MODELRUN_GOLDEN=model.yaml.
func TestRecallByDim(t *testing.T) {
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
	var keys, qkeys [][]float32
	for _, f := range recallFacts {
		keys = append(keys, m.keyOf(m.pieces(f[0])))
		qkeys = append(qkeys, m.keyOf(m.pieces(f[1])))
	}
	cos1, cos3 := 0, 0
	for i, q := range qkeys {
		type sc struct {
			j int
			s float64
		}
		var s []sc
		for j, k := range keys {
			var d float64
			for x := range k {
				d += float64(k[x]) * float64(q[x])
			}
			s = append(s, sc{j, d})
		}
		sort.Slice(s, func(a, b int) bool { return s[a].s > s[b].s })
		if s[0].j == i {
			cos1++
		}
		for _, x := range s[:3] {
			if x.j == i {
				cos3++
			}
		}
	}
	t.Logf("cosine ceiling        top-1 %2d/%d  top-3 %2d/%d", cos1, len(keys), cos3, len(keys))
	var texts []string
	for _, f := range recallFacts {
		texts = append(texts, f[0])
	}
	for _, dims := range []string{"64x64", "16x16x16", "8x8x8x8"} {
		t.Setenv("HES_DIMS", dims)
		h, err := m.libraryFor(texts)
		if err != nil {
			t.Fatal(err)
		}
		top1, top3 := 0, 0
		for i, f := range recallFacts {
			got := m.recallFor(h, f[1], nil)
			if len(got) > 0 && got[0].Payload == recallFacts[i][0] {
				top1++
			}
			for k := 0; k < 3 && k < len(got); k++ {
				if got[k].Payload == recallFacts[i][0] {
					top3++
				}
			}
		}
		t.Logf("HES %-12s top-1 %2d/%d  top-3 %2d/%d", fmt.Sprint(dims), top1, len(keys), top3, len(keys))
		if dims == "64x64" && (top1 < 27 || top3 < 30) {
			t.Errorf("default torus recall fell: top-1 %d (want ≥ 27), top-3 %d (want 30)", top1, top3)
		}
	}
}
