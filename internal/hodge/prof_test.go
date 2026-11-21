package hodge

import (
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"gptrun/internal/catalog"
	"gptrun/internal/rt"
)

// TestProfile times parts of one decode step on the real device.
// MODELRUN_PROF=path/to/model.yaml go test -run TestProfile -v ./internal/hodge
func TestProfile(t *testing.T) {
	cfg := os.Getenv("MODELRUN_PROF")
	if cfg == "" {
		t.Skip("MODELRUN_PROF is not set")
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
	d := &dialog{}
	ids := m.tok.Encode("The quick brown fox jumps over the lazy dog. Once upon a time", true)
	if err := m.prefill(d, ids); err != nil {
		t.Fatal(err)
	}
	if err := m.head(d, &headReq{exact: true}); err != nil {
		t.Fatal(err)
	}
	time1 := func(name string, reps int, f func() error) {
		var ts []time.Duration
		for i := 0; i < reps; i++ {
			t0 := time.Now()
			if err := f(); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			ts = append(ts, time.Since(t0))
		}
		sort.Slice(ts, func(i, j int) bool { return ts[i] < ts[j] })
		t.Logf("%-28s median %8.2f ms  min %8.2f ms", name, ms(ts[len(ts)/2]), ms(ts[0]))
	}
	run := func(ops []rt.Op, gets []rt.Get) func() error {
		return func() error { _, err := g.dev.Run(ops, gets); return err }
	}
	var linOnly, attnOps, normOps []rt.Op
	for li := range g.layers {
		L := &g.layers[li]
		for _, l := range [][]rt.Op{L.qkv, L.o, L.gateUp, L.down} {
			linOnly = append(linOnly, l...)
		}
		attnOps = append(attnOps, disp(g.ropeP, L.rope, g.ropeX)...)
		attnOps = append(attnOps, disp(g.attnP, L.attn, g.attnX)...)
		normOps = append(normOps, disp(g.normD, L.attnN, 1)...)
		normOps = append(normOps, disp(g.subP, L.sub, 1)...)
		normOps = append(normOps, disp(g.normD, L.ffnN, 1)...)
		normOps = append(normOps, disp(g.normF, L.ffnS, 1)...)
	}
	ch, _, _ := g.chunk(ids[0])
	layerOps := append(disp(g.embedP, ch.embedG, g.embedX), g.tok...)
	head := g.head
	one := disp(g.normD, g.layers[0].attnN, 1)
	logits := []rt.Get{{Buf: g.logits, N: m.Vocab * 4}}
	t.Logf("dispatches per token: %d layers + %d head", len(layerOps), len(head))
	time1("round trip (1 norm)", 20, run(one, nil))
	time1("layers", 8, run(layerOps, nil))
	time1("linears only", 8, run(linOnly, nil))
	time1("rope+attn only", 8, run(attnOps, nil))
	for _, pos := range []int{512, 1500, 3000} {
		rec := make([]byte, 16)
		binary.LittleEndian.PutUint32(rec, uint32(pos))
		if err := g.dev.Write("step", 0, rec); err != nil {
			t.Fatal(err)
		}
		time1(fmt.Sprintf("rope+attn at pos %d", pos), 6, run(attnOps, nil))
	}
	{
		rec := make([]byte, 16)
		binary.LittleEndian.PutUint32(rec, uint32(d.pos-1))
		_ = g.dev.Write("step", 0, rec)
	}
	time1("norms only", 8, run(normOps, nil))
	time1("head (no readback)", 8, run(head, nil))
	time1("logits readback", 8, run(one, logits))
	time1("full step + readback", 8, run(append(append([]rt.Op{}, layerOps...), head...), logits))
	lg := append([]float32(nil), d.lg...)
	recent := ids
	keep := func(id int) bool { return m.stop[id] }
	time1("sample greedy", 8, func() error {
		x := append([]float32(nil), lg...)
		draw(x, recent, 0, 0.9, keep, m.stopIDs(), nil)
		return nil
	})
	if sc := g.screen; sc != nil {
		scr := []rt.Op{g.head[0], {Tag: rt.OpDispatch, Pipe: sc.scrP, Group: sc.scrG, X: sc.scrX, Y: sc.scrY, Z: 1}}
		time1("screen kernel (no readback)", 8, run(scr, nil))
		time1("screen + ap,y readback", 8, run(scr, []rt.Get{{Buf: sc.ap, N: m.Vocab * 4}, {Buf: sc.y, N: m.D * 4}}))
		out, _ := g.dev.Run(scr, []rt.Get{{Buf: sc.ap, N: m.Vocab * 4}, {Buf: sc.y, N: m.D * 4}})
		time1("candidates (CPU) greedy", 8, func() error { m.candidates(out, &headReq{keep: keep}); return nil })
		time1("candidates (CPU) 0.6", 8, func() error {
			if !m.candidates(out, &headReq{temp: 0.6, keep: keep}) {
				t.Logf("0.6 fell back")
			}
			return nil
		})
	}
	greedy := &headReq{keep: keep}
	warm := &headReq{temp: 0.6, keep: keep}
	time1("head exact", 8, func() error { return m.head(d, &headReq{exact: true}) })
	time1("head screened greedy", 8, func() error { return m.head(d, greedy) })
	time1("head screened 0.6", 8, func() error { g.screen.cool = 0; return m.head(d, warm) })
	t.Logf("screen kept argmax %d  window %d  cut1 %.3f  trueBest %.3f", g.screen.nArgmax, g.screen.nWindow, g.screen.cut1, g.screen.trueBest)
	hist := make([]int, 8)
	below := []float64{0, 1, 2, 4, 8, 16, 32, 64}
	for _, u := range g.screen.ub {
		for i, w := range below {
			if u >= g.screen.cut1-w {
				hist[i]++
			}
		}
	}
	t.Logf("rows with ub within 0,1,2,4,8,16,32,64 of the argmax cut: %v", hist)
	time1("feed+head greedy", 8, func() error { return m.feed(d, ids[1], greedy) })
	time1("feed+head 0.6", 8, func() error { g.screen.cool = 0; return m.feed(d, ids[1], warm) })
	d2 := &dialog{}
	long := make([]int, 0, 128)
	for len(long) < 128 {
		long = append(long, ids...)
	}
	long = long[:128]
	sync := func() error { _, err := g.dev.Run(nil, nil); return err }
	if b := g.batch; b != nil {
		// One 32-token batch at position 0, by part.
		if err := m.stepB(d2, long[:32], nil); err != nil {
			t.Fatal(err)
		}
		d2.pos, d2.ids = 0, nil
		var lin, att, nrm []rt.Op
		for i := range b.layers {
			L := &b.layers[i]
			lin = append(lin, L.qkv, L.o, L.gu, L.down)
			att = append(att, rt.Op{Tag: rt.OpDispatch, Pipe: b.ropeP, Group: L.rope, X: b.ropeX, Y: 32, Z: 1},
				rt.Op{Tag: rt.OpDispatch, Pipe: b.attnP, Group: L.attn, X: b.attnX, Y: 32, Z: 1})
			nrm = append(nrm, rt.Op{Tag: rt.OpDispatch, Pipe: b.normD, Group: L.an, X: 32, Y: 1, Z: 1},
				rt.Op{Tag: rt.OpDispatch, Pipe: b.normA, Group: L.sub, X: 32, Y: 1, Z: 1},
				rt.Op{Tag: rt.OpDispatch, Pipe: b.normD, Group: L.fn, X: 32, Y: 1, Z: 1},
				rt.Op{Tag: rt.OpDispatch, Pipe: b.normF, Group: L.fs, X: 32, Y: 1, Z: 1})
		}
		time1("batch32 linears", 4, run(lin, nil))
		time1("batch32 qkv only", 4, run(func() []rt.Op { var o []rt.Op; for i := range b.layers { o = append(o, b.layers[i].qkv) }; return o }(), nil))
		time1("batch32 gate/up only", 4, run(func() []rt.Op { var o []rt.Op; for i := range b.layers { o = append(o, b.layers[i].gu) }; return o }(), nil))
		time1("batch32 down only", 4, run(func() []rt.Op { var o []rt.Op; for i := range b.layers { o = append(o, b.layers[i].down) }; return o }(), nil))
		time1("batch32 rope+attn", 4, run(att, nil))
		time1("batch32 norms", 4, run(nrm, nil))
		time1("batch32 full", 4, func() error {
			d2.pos, d2.ids = 0, nil
			if err := m.stepB(d2, long[:32], nil); err != nil {
				return err
			}
			return sync()
		})
	}
	time1("prefill 128 tokens", 3, func() error {
		d2.pos, d2.ids, g.kv = 0, nil, nil
		if err := m.prefill(d2, long); err != nil {
			return err
		}
		return sync()
	})
	_ = d
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
