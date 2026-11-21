package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"gptrun/internal/rt"
)

// The quaternary chart is the routed mixture. gate-up and down stay packed:
// [experts, rows/4, cols], limb 0's zero code is 1, and further limbs carry
// their own radix. The tensor shape is the grid. Logical rows are shape[1]*4.
// A shapeless BNZ plane falls back to the coarse index. The grids are not
// expanded to hidden × intermediate. A layer that also has gate_proj keeps
// that dense relu2 block as the shared expert and adds the chart beside it.

const (
	gptTopK  = 4
	gptAlpha = 1.702
	gptLimit = 7
)

type axisMap struct {
	n, g   int
	i0, i1 []int
	w0     []float32
}

type liftSet struct {
	d, f, two, dout axisMap
}

type gridW struct {
	rows, cols int
	w          []byte
	s          float32
	limbs      []limb
}

type moeFFN struct {
	experts   int
	topK      int
	D, F, two int
	alpha     float32
	limit     float32
	gu, down  gridW
	guBias    []float32 // [experts, 2F], gate and up interleaved
	dnBias    []float32 // [experts, D]
	router    []float32 // [experts, D]
	rbias     []float32
	lift      *liftSet
	sub       []float32 // ffn_sub_norm, length F. Nil is the unit gain the dense path skips.
	eps       float64
	gemma     bool
}

func gridSize(n int) int {
	g := 1
	for _, r := range radicesDesc(n) {
		if g*r > 256 {
			break
		}
		g *= r
	}
	if g < 1 {
		return 1
	}
	return g
}

func radicesDesc(n int) []int {
	var out []int
	m := n
	for p := 2; p*p <= m; p++ {
		for m%p == 0 {
			out = append(out, p)
			m /= p
		}
	}
	if m > 1 {
		out = append(out, m)
	}
	if len(out) == 0 {
		return []int{1}
	}
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// newAxisMap is sampleGrid's linear interpolation on linCoords, as gather
// indices. down is the adjoint (fine activation onto the grid). up scatters
// a grid vector back onto the fine axis.
func newAxisMap(n, g int) axisMap {
	a := axisMap{n: n, g: g, i0: make([]int, n), i1: make([]int, n), w0: make([]float32, n)}
	for i := 0; i < n; i++ {
		u := (float64(i)+0.5)/float64(n)*float64(g) - 0.5
		if u <= 0 || g <= 1 {
			a.i0[i], a.i1[i], a.w0[i] = 0, 0, 1
			continue
		}
		if u >= float64(g-1) {
			last := g - 1
			a.i0[i], a.i1[i], a.w0[i] = last, last, 1
			continue
		}
		k := int(u)
		a.i0[i] = k
		a.i1[i] = k + 1
		a.w0[i] = 1 - float32(u-float64(k))
	}
	return a
}

func (a axisMap) down(x []float32) []float32 {
	o := make([]float32, a.g)
	for i := 0; i < a.n && i < len(x); i++ {
		o[a.i0[i]] += a.w0[i] * x[i]
		if a.i1[i] != a.i0[i] {
			o[a.i1[i]] += (1 - a.w0[i]) * x[i]
		}
	}
	return o
}

func (a axisMap) up(c []float32) []float32 {
	o := make([]float32, a.n)
	for i := 0; i < a.n; i++ {
		o[i] = a.w0[i] * c[a.i0[i]]
		if a.i1[i] != a.i0[i] {
			o[i] += (1 - a.w0[i]) * c[a.i1[i]]
		}
	}
	return o
}

func liftsFor(D, F int) *liftSet {
	return liftsFrom(D, F, gridSize(D), gridSize(2*F), gridSize(F), gridSize(D))
}

// liftsFrom is the Whitney sample on the stored grids. d gathers the hidden
// onto the gate-up rows, two scatters the gate-up columns onto 2F, f gathers
// the SwiGLU hidden onto the down rows, and dout scatters the down columns
// back onto the hidden.
func liftsFrom(D, F, guRows, guCols, dnRows, dnCols int) *liftSet {
	return &liftSet{
		d:    newAxisMap(D, guRows),
		f:    newAxisMap(F, dnRows),
		two:  newAxisMap(2*F, guCols),
		dout: newAxisMap(D, dnCols),
	}
}

func shape3(shape []int, e, r, c int) bool {
	return len(shape) == 3 && shape[0] == e && shape[1] == r && shape[2] == c
}

// packedGrid is the quaternary plane. BNZ4 keeps a shape only for the rank-2
// tern and bf16 methods, so a rank-3 expert grid arrives as bytes alone.
func packedGrid(g got, name string, experts, rows, cols int) error {
	if rows%4 != 0 || cols < 1 || experts < 1 {
		return fmt.Errorf("%s: grid %d experts of %d×%d", name, experts, rows, cols)
	}
	np := rows / 4
	want := experts * np * cols
	if len(g.b) != want || (len(g.shape) != 0 && !shape3(g.shape, experts, np, cols)) {
		return fmt.Errorf("%s is %d bytes %v, want U8 [%d %d %d]", name, len(g.b), g.shape, experts, np, cols)
	}
	return nil
}

func loadGrid(t map[string]got, name string, experts, rows, cols int) (gridW, error) {
	g, err := need(t, name)
	if err != nil {
		return gridW{}, err
	}
	if err := packedGrid(g, name, experts, rows, cols); err != nil {
		return gridW{}, err
	}
	sc, err := need(t, name+"_scale")
	if err != nil {
		return gridW{}, err
	}
	s, err := bf16one(sc.b)
	if err != nil {
		return gridW{}, fmt.Errorf("%s: %w", name, err)
	}
	w := gridW{rows: rows, cols: cols, w: g.b, s: s}
	for i := 1; ; i++ {
		ln := fmt.Sprintf("%s_limb%d", name, i)
		lb, ok := t[ln]
		if !ok {
			break
		}
		if err := packedGrid(lb, ln, experts, rows, cols); err != nil {
			return gridW{}, err
		}
		ss, err := need(t, ln+"_scale")
		if err != nil {
			return gridW{}, err
		}
		sf, err := bf16one(ss.b)
		if err != nil {
			return gridW{}, err
		}
		rb, err := need(t, ln+"_radix")
		if err != nil {
			return gridW{}, err
		}
		if len(rb.b) < 4 {
			return gridW{}, fmt.Errorf("%s_radix is %d bytes", ln, len(rb.b))
		}
		radix := binary.LittleEndian.Uint32(rb.b[:4])
		// A residual limb that did not move is stored with scale 0 and radix 0.
		if sf == 0 {
			continue
		}
		if radix < 2 {
			return gridW{}, fmt.Errorf("%s radix %d", ln, radix)
		}
		w.limbs = append(w.limbs, limb{w: lb.b, s: sf, center: int((radix - 1) / 2)})
	}
	return w, nil
}

func f32n(t map[string]got, name string, n int) ([]float32, error) {
	g, err := need(t, name)
	if err != nil {
		return nil, err
	}
	w, err := bf16s(g.b)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if len(w) != n {
		return nil, fmt.Errorf("%s has %d values, want %d", name, len(w), n)
	}
	return w, nil
}

// loadMoE reads one layer's packed expert grids. lifts is shared across layers.
func loadMoE(t map[string]got, prefix string, D, F, topK int, alpha, limit float32, lifts *liftSet, expertsWant int) (*moeFFN, *liftSet, error) {
	guName := prefix + "mlp.experts.gate_up_proj"
	dnName := prefix + "mlp.experts.down_proj"
	guT, err := need(t, guName)
	if err != nil {
		return nil, lifts, err
	}
	guRows, guCols := gridSize(D), gridSize(2*F)
	dnRows, dnCols := gridSize(F), gridSize(D)
	experts := expertsWant
	switch len(guT.shape) {
	case 3:
		experts = guT.shape[0]
		if guT.shape[1] < 1 || guT.shape[2] < 1 {
			return nil, lifts, fmt.Errorf("%s shape %v", guName, guT.shape)
		}
		guRows, guCols = guT.shape[1]*4, guT.shape[2]
	case 0:
		if experts < 1 {
			return nil, lifts, fmt.Errorf("%s has no shape; config num_local_experts is not set", guName)
		}
	default:
		return nil, lifts, fmt.Errorf("%s shape %v", guName, guT.shape)
	}
	if dnT, ok := t[dnName]; ok && len(dnT.shape) == 3 && dnT.shape[0] == experts && dnT.shape[1] > 0 && dnT.shape[2] > 0 {
		dnRows, dnCols = dnT.shape[1]*4, dnT.shape[2]
	}
	if expertsWant > 0 && experts != expertsWant {
		return nil, lifts, fmt.Errorf("%s has %d experts, config says %d", guName, experts, expertsWant)
	}
	if topK < 1 || topK > experts {
		return nil, lifts, fmt.Errorf("%s: top-%d of %d experts", guName, topK, experts)
	}
	gu, err := loadGrid(t, guName, experts, guRows, guCols)
	if err != nil {
		return nil, lifts, err
	}
	dn, err := loadGrid(t, dnName, experts, dnRows, dnCols)
	if err != nil {
		return nil, lifts, err
	}
	two := 2 * F
	gb, err := f32n(t, guName+"_bias", experts*two)
	if err != nil {
		return nil, lifts, err
	}
	db, err := f32n(t, dnName+"_bias", experts*D)
	if err != nil {
		return nil, lifts, err
	}
	rw, err := f32n(t, prefix+"mlp.router.weight", experts*D)
	if err != nil {
		return nil, lifts, err
	}
	rb, err := f32n(t, prefix+"mlp.router.bias", experts)
	if err != nil {
		return nil, lifts, err
	}
	lifts = liftsFrom(D, F, gu.rows, gu.cols, dn.rows, dn.cols)
	return &moeFFN{
		experts: experts, topK: topK, D: D, F: F, two: two,
		alpha: alpha, limit: limit,
		gu: gu, down: dn, guBias: gb, dnBias: db, router: rw, rbias: rb,
		lift: lifts,
	}, lifts, nil
}

// applyCol is y[c] = sum_r W[r,c] x[r]. Byte r of packed column c holds the
// 2-bit codes of rows r, r+NP, r+2NP, r+3NP. Limb 0's zero code is 1.
func (w gridW) applyCol(e int, x []float32) []float32 {
	y := make([]float32, w.cols)
	np := w.rows / 4
	plane := np * w.cols
	add := func(buf []byte, scale float32, center int) {
		b := buf[e*plane : (e+1)*plane]
		for r := 0; r < np; r++ {
			row := b[r*w.cols : (r+1)*w.cols]
			x0 := x[r]
			x1 := x[r+np]
			x2 := x[r+2*np]
			x3 := x[r+3*np]
			for c := 0; c < w.cols; c++ {
				by := row[c]
				y[c] += (float32(int((by)&3)-center)*x0 +
					float32(int((by>>2)&3)-center)*x1 +
					float32(int((by>>4)&3)-center)*x2 +
					float32(int((by>>6)&3)-center)*x3) * scale
			}
		}
	}
	add(w.w, w.s, 1)
	for i := range w.limbs {
		lb := &w.limbs[i]
		add(lb.w, lb.s, lb.center)
	}
	return y
}

func (f *moeFFN) delta(x []float32) []float32 {
	logits := make([]float32, f.experts)
	for e := 0; e < f.experts; e++ {
		row := f.router[e*f.D : (e+1)*f.D]
		var s float32
		for i, v := range x {
			s += row[i] * v
		}
		logits[e] = s + f.rbias[e]
	}
	idx, wt := moeTopK(logits, f.topK)
	xg := f.lift.d.down(x)
	z := make([]float32, f.D)
	g := make([]float32, f.F)
	u := make([]float32, f.F)
	for t, e := range idx {
		w := wt[t]
		fine := f.lift.two.up(f.gu.applyCol(e, xg))
		bias := f.guBias[e*f.two : (e+1)*f.two]
		for r := 0; r < f.F; r++ {
			g[r] = fine[2*r] + bias[2*r]
			u[r] = fine[2*r+1] + bias[2*r+1]
		}
		h := moeSwiGLU(g, u, f.alpha, f.limit)
		if len(f.sub) == len(h) && !unit(f.sub) {
			h = rms(h, f.sub, f.eps, f.gemma)
		}
		yd := f.down.applyCol(e, f.lift.f.down(h))
		out := f.lift.dout.up(yd)
		db := f.dnBias[e*f.D : (e+1)*f.D]
		for i := range z {
			z[i] += w * (out[i] + db[i])
		}
	}
	return z
}

func moeSwiGLU(gate, up []float32, alpha, limit float32) []float32 {
	h := make([]float32, len(gate))
	for i := range h {
		g, u := gate[i], up[i]
		if g > limit {
			g = limit
		}
		if u > limit {
			u = limit
		} else if u < -limit {
			u = -limit
		}
		sig := float32(1 / (1 + math.Exp(float64(-alpha*g))))
		h[i] = (u + 1) * (g * sig)
	}
	return h
}

func vecRms(v []float32) float64 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if len(v) == 0 {
		return 0
	}
	return math.Sqrt(s / float64(len(v)))
}

func vecRmsDiff(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var s float64
	for i := 0; i < n; i++ {
		d := float64(a[i] - b[i])
		s += d * d
	}
	if n == 0 {
		return 0
	}
	return math.Sqrt(s / float64(n))
}

func moeTopK(logits []float32, k int) ([]int, []float32) {
	if k > len(logits) {
		k = len(logits)
	}
	idx := make([]int, k)
	used := make([]bool, len(logits))
	for t := 0; t < k; t++ {
		best, bi := float32(-1e30), 0
		for i, v := range logits {
			if used[i] {
				continue
			}
			if v > best || (v == best && i < bi) {
				best, bi = v, i
			}
		}
		used[bi] = true
		idx[t] = bi
	}
	mx := logits[idx[0]]
	for _, i := range idx[1:] {
		if logits[i] > mx {
			mx = logits[i]
		}
	}
	w := make([]float32, k)
	var z float32
	for t, i := range idx {
		w[t] = float32(math.Exp(float64(logits[i] - mx)))
		z += w[t]
	}
	for t := range w {
		w[t] /= z
	}
	return idx, w
}

func (g *gpuRT) moePre(i int) {
	L := &g.layers[i]
	ops := make([]rt.Op, 0, 16)
	ops = append(ops, disp(g.normD, L.attnN, 1)...)
	ops = append(ops, L.qkv...)
	ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: g.ropeP, Group: L.rope, X: g.ropeX, Y: 1, Z: 1})
	ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: g.attnP, Group: L.attn, X: g.attnX, Y: 1, Z: 1})
	ops = append(ops, disp(g.subP, L.sub, 1)...)
	ops = append(ops, L.o...)
	ops = append(ops, disp(g.normD, L.ffnN, 1)...)
	L.pre = ops
}

// stepMoE runs attention on the device and the quaternary chart on the host.
// One token at a time: the next token's attention reads this token's cache.
// A layer that also has the dense block adds that block on the device. The
// chart is the routed residual on the same normalized activation.
func (m *Model) stepMoE(d *dialog, ids []int, req *headReq) error {
	g := m.gpu
	if len(ids) == 0 {
		return nil
	}
	if len(ids) > maxBatch {
		return fmt.Errorf("step: %d tokens in one submit", len(ids))
	}
	if d.pos+len(ids) > g.ctx {
		return fmt.Errorf("context full (%d)", g.ctx)
	}
	pos0 := d.pos
	g.ran(d, ids)
	y := make([]float32, m.D)
	rec := make([]byte, g.stride)
	for t, id := range ids {
		if interrupted.Load() {
			return ErrInterrupted
		}
		ch, row, err := g.chunk(id)
		if err != nil {
			return err
		}
		clear(rec)
		binary.LittleEndian.PutUint32(rec[0:], uint32(pos0+t))
		binary.LittleEndian.PutUint32(rec[4:], uint32(row))
		var wr []byte
		for li := range m.layers {
			ff := m.layers[li].moe
			if ff == nil {
				return fmt.Errorf("layer %d has no experts", li)
			}
			L := &g.layers[li]
			ops := make([]rt.Op, 0, 4+len(L.pre))
			writes := []rt.Write{{Buf: g.stepID, Data: rec}}
			if li == 0 {
				ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: g.embedP, Group: ch.embedG, X: g.embedX, Y: 1, Z: 1})
			} else if len(wr) > 0 {
				writes = append(writes, rt.Write{Buf: g.wrBuf, Data: wr})
				ops = append(ops, disp(g.addP, g.addG, g.addX)...)
				wr = nil
			}
			ops = append(ops, L.pre...)
			watch := req != nil && t == len(ids)-1 && (li == 0 || li == 29 || li == 35)
			gets := []rt.Get{{Buf: g.yBuf, N: m.D * 4}}
			if watch {
				gets = append(gets, rt.Get{Buf: g.xBuf, N: m.D * 4})
			}
			out, err := g.dev.Run2(writes, ops, gets, true)
			if err != nil {
				return err
			}
			if len(out) < m.D*4 {
				return fmt.Errorf("layer %d activation is %d bytes", li, len(out))
			}
			y = decodeF32(y, out, m.D)
			var xBefore []float32
			if watch {
				xBefore = decodeF32(nil, out[m.D*4:], m.D)
			}
			z := ff.delta(y)
			if len(L.gateUp) == 0 {
				if watch {
					logf("L%d y %.3g chart %.3g  no dense", li, vecRms(y), vecRms(z))
				}
				wr = f32b(z)
				continue
			}
			// Chart first, then the dense block. The down residual writes wr
			// after the chart add has already been dispatched.
			ffn := make([]rt.Op, 0, 8+len(L.gateUp)+len(L.down))
			writes = []rt.Write{{Buf: g.wrBuf, Data: f32b(z)}}
			if os.Getenv("HODGE_NOCHART") != "1" {
				ffn = append(ffn, disp(g.addP, g.addG, g.addX)...)
			} else {
				writes = nil
			}
			ffn = append(ffn, L.gateUp...)
			ffn = append(ffn, disp(g.normF, L.ffnS, 1)...)
			ffn = append(ffn, L.down...)
			gets = nil
			if watch {
				gets = []rt.Get{{Buf: g.xBuf, N: m.D * 4}}
			}
			got, err := g.dev.Run2(writes, ffn, gets, true)
			if err != nil {
				return err
			}
			if watch && len(got) >= m.D*4 {
				xAfter := decodeF32(nil, got, m.D)
				logf("L%d y %.3g chart %.3g  x %.3g -> %.3g  move %.3g", li, vecRms(y), vecRms(z), vecRms(xBefore), vecRms(xAfter), vecRmsDiff(xBefore, xAfter))
			}
		}
		if t == len(ids)-1 && req != nil {
			if len(wr) > 0 {
				return m.runHead(d, []rt.Write{{Buf: g.wrBuf, Data: wr}}, disp(g.addP, g.addG, g.addX), req)
			}
			return m.runHead(d, nil, nil, req)
		}
		if len(wr) == 0 {
			continue
		}
		if _, err := g.dev.Run2([]rt.Write{{Buf: g.wrBuf, Data: wr}}, disp(g.addP, g.addG, g.addX), nil, true); err != nil {
			return err
		}
	}
	return nil
}

// stepMix is a chart whose base layers keep the dense block and whose added
// layers are the expert pack. Attention stays on the device. A dense layer
// finishes there. An expert layer reads the activation and adds the host grid.
func (m *Model) stepMix(d *dialog, ids []int, req *headReq) error {
	g := m.gpu
	if len(ids) == 0 {
		return nil
	}
	if len(ids) > maxBatch {
		return fmt.Errorf("step: %d tokens in one submit", len(ids))
	}
	if d.pos+len(ids) > g.ctx {
		return fmt.Errorf("context full (%d)", g.ctx)
	}
	pos0 := d.pos
	g.ran(d, ids)
	y := make([]float32, m.D)
	rec := make([]byte, g.stride)
	for t, id := range ids {
		if interrupted.Load() {
			return ErrInterrupted
		}
		ch, row, err := g.chunk(id)
		if err != nil {
			return err
		}
		clear(rec)
		binary.LittleEndian.PutUint32(rec[0:], uint32(pos0+t))
		binary.LittleEndian.PutUint32(rec[4:], uint32(row))
		var wr []byte
		for li := range m.layers {
			L := &m.layers[li]
			gl := &g.layers[li]
			ops := make([]rt.Op, 0, 8+len(gl.pre))
			writes := []rt.Write{{Buf: g.stepID, Data: rec}}
			if li == 0 {
				ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: g.embedP, Group: ch.embedG, X: g.embedX, Y: 1, Z: 1})
			} else if wr != nil {
				writes = append(writes, rt.Write{Buf: g.wrBuf, Data: wr})
				ops = append(ops, disp(g.addP, g.addG, g.addX)...)
			}
			ops = append(ops, gl.pre...)
			if L.moe != nil {
				out, err := g.dev.Run2(writes, ops, []rt.Get{{Buf: g.yBuf, N: m.D * 4}}, true)
				if err != nil {
					return err
				}
				if len(out) < m.D*4 {
					return fmt.Errorf("layer %d activation is %d bytes", li, len(out))
				}
				y = decodeF32(y, out, m.D)
				wr = f32b(L.moe.delta(y))
				continue
			}
			ops = append(ops, gl.gateUp...)
			ops = append(ops, disp(g.normF, gl.ffnS, 1)...)
			ops = append(ops, gl.down...)
			if _, err := g.dev.Run2(writes, ops, nil, true); err != nil {
				return err
			}
			wr = nil
		}
		if t == len(ids)-1 && req != nil {
			if wr != nil {
				return m.runHead(d, []rt.Write{{Buf: g.wrBuf, Data: wr}}, disp(g.addP, g.addG, g.addX), req)
			}
			return m.runHead(d, nil, nil, req)
		}
		if wr == nil {
			continue
		}
		if _, err := g.dev.Run2([]rt.Write{{Buf: g.wrBuf, Data: wr}}, disp(g.addP, g.addG, g.addX), nil, true); err != nil {
			return err
		}
	}
	return nil
}
