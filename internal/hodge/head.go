package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"gptrun/internal/rt"
)

// The LM head is the tied embedding, vocab x hidden bf16. With a large
// vocabulary it is most of the bytes a token reads, so the head runs in two
// passes that together give the sampler what the full head gives it:
//
//  1. lmscreen.wgsl reads an int8 copy (half the bytes) and writes ap[r].
//     |ap[r] - logit[r]| <= e[r] = ||y||_1 * (d[r] + slack*s[r]), where s[r] is
//     the row's scale and d[r] its largest rounding error (lmquant.wgsl).
//  2. The host keeps every row whose upper bound ap+e reaches the lowest the
//     sampler's cut can be: the best lower bound (after the repetition
//     penalty) for a greedy pick, that minus 20 temperatures for a draw (dist's
//     window). lmgather.wgsl computes those rows exactly, with lmhead.wgsl's
//     arithmetic. Every other row is below the cut, so the sampler never reads
//     it; it is -Inf in the logits handed over.
//
// The sampler window is one exact gather when those rows fit in the buffer.
// That is the common path, and it is the same set the one-pass cut kept.
// When that set does not fit, the head rescores the rows that can still be
// the argmax, reads the true best, and gathers the temperature window around
// that best when the window is narrower and fits. Rows outside it cannot be
// drawn. When a chunk still keeps more rows than the gather buffer, the full
// head runs, and keeps running for a few tokens before the screen is tried
// again. MODELRUN_HEAD=exact always runs the full head.

// headReq is what the sampler will do with the next logits.
type headReq struct {
	temp   float64 // <= 0: greedy
	recent []int   // the repetition penalty's window
	keep   func(int) bool
	exact  bool // every row exactly
}

// screenSlack covers the f32 rounding of both dot products, relative to
// ||y||_1 * s: each is under 1e-3 for a 4096-wide row. This is 60 times that.
const screenSlack = 0.0625

const (
	maxCand = 1 << 16 // candidate slots per embedding chunk
	// A screen that cannot fit the sampler window falls back to the full head
	// for that token only. Holding the full head for a fixed run made the
	// following peaked tokens read the bf16 embedding too: on hodge-5o at
	// temperature 0.6 a 32-token reply went from 2.72s to 3.14s.
	screenCool = 0
)

type lmScreen struct {
	sd, ap, cand, cnt, ex, y uint32
	scrP, scrG, scrX, scrY   uint32
	gather                   []gatherChunk
	s, dl                    []float32
	cool                     int
	// per-row scratch
	ub   []float64
	pen  []bool
	mark []bool
	part [][]uint32
	lg   []float32
	list [][]uint32
	more [][]uint32
	recs []byte
	// nArgmax, nWindow are the rows the last token rescored.
	// cut1 is the argmax cut, trueBest the rescored best.
	nArgmax, nWindow int
	cut1, trueBest   float64
}

type gatherChunk struct {
	p, g uint32
}

func (g *gpuRT) bootScreen(m *Model) error {
	if os.Getenv("MODELRUN_HEAD") == "exact" || m.D%16 != 0 || m.Vocab*m.D > maxStorage {
		return nil
	}
	d := g.dev
	V, D := m.Vocab, m.D
	quant, err := ownKernel("lmquant.wgsl")
	if err != nil {
		return err
	}
	screen, err := ownKernel("lmscreen.wgsl")
	if err != nil {
		return err
	}
	gather, err := ownKernel("lmgather.wgsl")
	if err != nil {
		return err
	}
	for _, s := range []struct{ name, src string }{{"lmquant", quant}, {"lmscreen", screen}, {"lmgather", gather}} {
		if err := d.Shader(s.name, s.src); err != nil {
			return err
		}
	}
	sc := &lmScreen{}
	if _, err := d.Buf("q8", nil, V*D); err != nil {
		return err
	}
	if sc.sd, err = d.Buf("sd", nil, V*8); err != nil {
		return err
	}
	if sc.ap, err = d.Buf("ap", nil, V*4); err != nil {
		return err
	}
	nc := len(g.chunks)
	if sc.cand, err = d.Buf("cand", nil, nc*maxCand*4); err != nil {
		return err
	}
	if sc.cnt, err = d.Buf("cnt", nil, 16*((nc+3)/4)); err != nil {
		return err
	}
	if sc.ex, err = d.Buf("ex", nil, nc*maxCand*4); err != nil {
		return err
	}
	if sc.y, err = d.Buf("y", nil, 16); err != nil {
		return err
	}
	var ops []rt.Op
	for i := range g.chunks {
		c := &g.chunks[i]
		name := fmt.Sprintf("lmq%d", c.start)
		p, err := d.Pipe(name, "lmquant", "main", fmt.Sprintf("D=%d ROW0=%d ROWS=%d", D, c.start, c.rows),
			[]byte{rt.KindRO, rt.KindRW, rt.KindRW})
		if err != nil {
			return err
		}
		gid, err := d.Group(name, name, []rt.Bind{{Buf: fmt.Sprintf("emb%d", c.start)}, {Buf: "q8"}, {Buf: "sd"}})
		if err != nil {
			return err
		}
		x, y := grid(uint32(c.rows))
		ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: p, Group: gid, X: x, Y: y, Z: 1})

		gname := fmt.Sprintf("lmg%d", c.start)
		gp, err := d.Pipe(gname, "lmgather", "main", fmt.Sprintf("D=%d CB=%d CI=%d", D, i*maxCand, i),
			[]byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW})
		if err != nil {
			return err
		}
		gg, err := d.Group(gname, gname, []rt.Bind{
			{Buf: "y"}, {Buf: fmt.Sprintf("emb%d", c.start)}, {Buf: "cand"}, {Buf: "cnt"}, {Buf: "ex"},
		})
		if err != nil {
			return err
		}
		sc.gather = append(sc.gather, gatherChunk{p: gp, g: gg})
	}
	out, err := d.Run(ops, []rt.Get{{Buf: sc.sd, N: V * 8}})
	if err != nil {
		return err
	}
	sc.s = make([]float32, V)
	sc.dl = make([]float32, V)
	for r := range V {
		sc.s[r] = math.Float32frombits(binary.LittleEndian.Uint32(out[r*8:]))
		sc.dl[r] = math.Float32frombits(binary.LittleEndian.Uint32(out[r*8+4:]))
	}
	if sc.scrP, err = d.Pipe("lmscreen", "lmscreen", "main", fmt.Sprintf("D=%d ROWS=%d", D, V),
		[]byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW}); err != nil {
		return err
	}
	if sc.scrG, err = d.Group("lmscreen", "lmscreen", []rt.Bind{{Buf: "y"}, {Buf: "q8"}, {Buf: "sd"}, {Buf: "ap"}}); err != nil {
		return err
	}
	sc.scrX, sc.scrY = grid(uint32((V + 63) / 64)) // lmscreen: 64 rows per workgroup
	sc.ub = make([]float64, V)
	sc.pen = make([]bool, V)
	sc.mark = make([]bool, V)
	sc.list = make([][]uint32, nc)
	sc.more = make([][]uint32, nc)
	g.screen = sc
	logf("head screen  int8 %d bytes", V*D)
	return nil
}

// grid spreads n workgroups over x and y, x at most 65535.
func grid(n uint32) (x, y uint32) {
	x, y = n, 1
	for x > 65535 {
		y *= 2
		x = (n + y - 1) / y
	}
	return x, y
}

// runHead appends the head to ops (the token's layers, or nothing), submits,
// and leaves the sampler's logits in d.lg.
func (m *Model) runHead(d *dialog, writes []rt.Write, ops []rt.Op, req *headReq) error {
	g := m.gpu
	sc := g.screen
	if sc == nil || req.exact || sc.cool > 0 {
		if sc != nil && sc.cool > 0 {
			sc.cool--
		}
		ops = append(ops, g.head...)
		out, err := g.dev.Run2(writes, ops, []rt.Get{{Buf: g.logits, N: m.Vocab * 4}}, false)
		if err != nil {
			return err
		}
		d.lg = decodeF32(d.lg, out, m.Vocab)
		return nil
	}
	ops = append(ops, g.head[0]) // the final norm: y
	ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: sc.scrP, Group: sc.scrG, X: sc.scrX, Y: sc.scrY, Z: 1})
	out, err := g.dev.Run2(writes, ops, []rt.Get{{Buf: sc.ap, N: m.Vocab * 4}, {Buf: sc.y, N: m.D * 4}}, false)
	if err != nil {
		return err
	}
	best, marked, ok := m.screenBest(out, req)
	defer sc.clearPen(marked)
	if !ok {
		return m.denseHead(d)
	}
	// The argmax set is small. Its exact best, not the screen's lower bound,
	// is what the sampler window is measured from, so a flat token gathers
	// the rows that can enter the sample instead of every row the bound
	// cannot reject.
	cut1 := cutUnder(best)
	sc.cut1 = cut1
	if !m.keepRows(cut1, sc.list, nil) {
		sc.nArgmax, sc.nWindow = rowCount(sc.list), 0
		return m.denseHead(d)
	}
	sc.nArgmax = rowCount(sc.list)
	if err := m.rescore(d, sc.list, true); err != nil {
		return err
	}
	sc.nWindow = sc.nArgmax
	if req.temp <= 0 {
		return nil
	}
	sc.trueBest = windowBest(d.lg, sc.pen)
	cut2, wider := widerCut(cut1, sc.trueBest, req.temp)
	if !wider {
		return nil
	}
	m.markKept(sc.list)
	defer m.unmarkKept(sc.list)
	if !m.keepRows(cut2, sc.more, sc.mark) {
		sc.nWindow = rowCount(sc.more)
		return m.denseHead(d)
	}
	sc.nWindow = sc.nArgmax + rowCount(sc.more)
	return m.rescore(d, sc.more, false)
}

func rowCount(lists [][]uint32) int {
	n := 0
	for _, l := range lists {
		n += len(l)
	}
	return n
}

// denseHead is the tied bf16 head. The screen is left off for a few tokens:
// the distribution was too flat for the gather buffer.
func (m *Model) denseHead(d *dialog) error {
	g := m.gpu
	g.screen.cool = screenCool
	var full []rt.Op
	for i := range g.chunks {
		c := &g.chunks[i]
		full = append(full, rt.Op{Tag: rt.OpDispatch, Pipe: c.lmP, Group: c.lmG, X: c.lmX, Y: c.lmY, Z: 1})
	}
	out, err := g.dev.Run2(nil, full, []rt.Get{{Buf: g.logits, N: m.Vocab * 4}}, false)
	if err != nil {
		return err
	}
	d.lg = decodeF32(d.lg, out, m.Vocab)
	return nil
}

// rescore writes the exact bf16 logit of each listed row into d.lg.
// reset fills every other row with -Inf. The lists are per-chunk local rows.
func (m *Model) rescore(d *dialog, lists [][]uint32, reset bool) error {
	g := m.gpu
	sc := g.screen
	nc := len(g.chunks)
	need := 0
	for _, l := range lists {
		need += len(l) * 4
	}
	recs := sc.recs[:0]
	if cap(recs) < need {
		recs = make([]byte, 0, need)
	}
	cnt := make([]byte, 16*((nc+3)/4))
	writes := make([]rt.Write, 0, nc+1)
	var gops []rt.Op
	var gets []rt.Get
	for i := range g.chunks {
		l := lists[i]
		binary.LittleEndian.PutUint32(cnt[i*4:], uint32(len(l)))
		if len(l) == 0 {
			continue
		}
		start := len(recs)
		for _, r := range l {
			recs = binary.LittleEndian.AppendUint32(recs, r)
		}
		writes = append(writes, rt.Write{Buf: sc.cand, Off: i * maxCand * 4, Data: recs[start:]})
		x, y := grid(uint32((len(l) + 7) / 8))
		gops = append(gops, rt.Op{Tag: rt.OpDispatch, Pipe: sc.gather[i].p, Group: sc.gather[i].g, X: x, Y: y, Z: 1})
		gets = append(gets, rt.Get{Buf: sc.ex, Off: i * maxCand * 4, N: len(l) * 4})
	}
	sc.recs = recs
	ninf := float32(math.Inf(-1))
	if len(d.lg) != m.Vocab {
		d.lg = make([]float32, m.Vocab)
		reset = true
	}
	if reset {
		for i := range d.lg {
			d.lg[i] = ninf
		}
	}
	if len(gops) == 0 {
		return nil
	}
	writes = append(writes, rt.Write{Buf: sc.cnt, Data: cnt})
	ex, err := g.dev.Run2(writes, gops, gets, false)
	if err != nil {
		return err
	}
	at := 0
	for i := range g.chunks {
		base := g.chunks[i].start
		for _, r := range lists[i] {
			d.lg[base+int(r)] = math.Float32frombits(binary.LittleEndian.Uint32(ex[at:]))
			at += 4
		}
	}
	return nil
}

// candidates reads the screen (ap, then y) and fills sc.list with the rows,
// per chunk, that the sampler's cut cannot rule out. False is the full head.
// The rows are split over the cores; the lists keep row order.
func (m *Model) candidates(out []byte, req *headReq) bool {
	best, marked, ok := m.screenBest(out, req)
	defer m.gpu.screen.clearPen(marked)
	if !ok {
		return false
	}
	cut := cutUnder(best)
	if req.temp > 0 {
		cut -= 20 * req.temp
	}
	return m.keepRows(cut, m.gpu.screen.list, nil)
}

// screenBest fills sc.ub from the screen readback (ap, then y) and returns
// the best penalized lower bound. marked are the repetition-penalty ids it
// set; the caller clears them. ok is false when a value is not a logit.
func (m *Model) screenBest(out []byte, req *headReq) (best float64, marked []int, ok bool) {
	g := m.gpu
	sc := g.screen
	V, D := m.Vocab, m.D
	y1 := 0.0
	for j := range D {
		v := float64(math.Float32frombits(binary.LittleEndian.Uint32(out[V*4+j*4:])))
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, nil, false
		}
		y1 += math.Abs(v)
	}
	y1 *= 1 + 1e-9
	pen := sc.pen
	for _, id := range req.recent {
		if id >= 0 && id < V && !pen[id] && (req.keep == nil || !req.keep(id)) {
			pen[id] = true
			marked = append(marked, id)
		}
	}
	P := min(runtime.GOMAXPROCS(0), 8)
	span := (V + P - 1) / P
	bests := make([]float64, P)
	bad := make([]bool, P)
	ub := sc.ub
	var wg sync.WaitGroup
	for p := range P {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			b := math.Inf(-1)
			for r := p * span; r < min(V, (p+1)*span); r++ {
				a := float64(math.Float32frombits(binary.LittleEndian.Uint32(out[r*4:])))
				if math.IsNaN(a) || math.IsInf(a, 0) {
					bad[p] = true
					return
				}
				e := y1 * (float64(sc.dl[r]) + screenSlack*float64(sc.s[r]))
				ub[r] = a + e
				lo := a - e
				if pen[r] {
					lo = penalized(lo)
				}
				b = max(b, lo)
			}
			bests[p] = b
		}(p)
	}
	wg.Wait()
	best = math.Inf(-1)
	for p := range P {
		if bad[p] {
			return 0, marked, false
		}
		best = max(best, bests[p])
	}
	return best, marked, true
}

func (sc *lmScreen) clearPen(marked []int) {
	for _, id := range marked {
		sc.pen[id] = false
	}
}

// cutUnder is a hair under best, so a logit that rounds to best still qualifies.
func cutUnder(best float64) float64 {
	return best - 1e-5*(1+math.Abs(best))
}

// widerCut is the sampler floor around the rescored best. wider is set when
// that floor sits below the argmax cut and more rows have to be rescored.
func widerCut(argmaxCut, trueBest, temp float64) (cut float64, wider bool) {
	if temp <= 0 || math.IsInf(trueBest, -1) {
		return argmaxCut, false
	}
	cut = trueBest - 20*temp - 1e-5*(1+math.Abs(trueBest))
	return cut, cut < argmaxCut
}

// windowBest is the sampler's best logit: the max exact value after the penalty.
func windowBest(lg []float32, pen []bool) float64 {
	best := math.Inf(-1)
	for i, v := range lg {
		x := float64(v)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			continue
		}
		if i < len(pen) && pen[i] {
			x = penalized(x)
		}
		best = max(best, x)
	}
	return best
}

// keepRows appends into dst, per chunk, every row whose upper bound reaches
// cut, skipping ids marked in skip. False means a chunk filled the gather buffer.
// Row order matches the vocabulary. skip may be nil.
func (m *Model) keepRows(cut float64, dst [][]uint32, skip []bool) bool {
	g := m.gpu
	sc := g.screen
	V := m.Vocab
	P := min(runtime.GOMAXPROCS(0), 8)
	span := (V + P - 1) / P
	if len(sc.part) != P {
		sc.part = make([][]uint32, P)
	}
	ub := sc.ub
	var wg sync.WaitGroup
	for p := range P {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			l := sc.part[p][:0]
			for r := p * span; r < min(V, (p+1)*span); r++ {
				if skip != nil && skip[r] {
					continue
				}
				if ub[r] >= cut {
					l = append(l, uint32(r))
					if len(l) > maxCand {
						break
					}
				}
			}
			sc.part[p] = l
		}(p)
	}
	wg.Wait()
	for i := range dst {
		dst[i] = dst[i][:0]
	}
	ci := 0
	for _, l := range sc.part {
		for _, r32 := range l {
			r := int(r32)
			for r >= g.chunks[ci].start+g.chunks[ci].rows {
				ci++
			}
			if len(dst[ci]) == maxCand {
				return false
			}
			dst[ci] = append(dst[ci], uint32(r-g.chunks[ci].start))
		}
	}
	return true
}

func (m *Model) markKept(lists [][]uint32) {
	g := m.gpu
	mark := g.screen.mark
	for i := range g.chunks {
		base := g.chunks[i].start
		for _, r := range lists[i] {
			mark[base+int(r)] = true
		}
	}
}

func (m *Model) unmarkKept(lists [][]uint32) {
	g := m.gpu
	mark := g.screen.mark
	for i := range g.chunks {
		base := g.chunks[i].start
		for _, r := range lists[i] {
			mark[base+int(r)] = false
		}
	}
}

// penalized is penalize's change to one logit, in float64.
func penalized(v float64) float64 {
	if v > 0 {
		return v / repeatPen
	}
	return v * repeatPen
}

func decodeF32(dst []float32, b []byte, n int) []float32 {
	if len(dst) != n {
		dst = make([]float32, n)
	}
	for j := range dst {
		dst[j] = math.Float32frombits(binary.LittleEndian.Uint32(b[j*4:]))
	}
	return dst
}
