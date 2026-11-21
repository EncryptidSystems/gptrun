package hodge

import (
	"embed"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"gptrun/internal/paths"
	"gptrun/internal/rt"
)

// ModelRun's own kernels: the fused HodgeLinear and the residual add. The rest
// load from the kernels directory as written.
//
//go:embed kernels/*.wgsl
var ownKernels embed.FS

func ownKernel(name string) (string, error) {
	b, err := ownKernels.ReadFile("kernels/" + name)
	return string(b), err
}

// maxBatch is how many tokens one submit carries. Each token has its own step
// record at a dynamic uniform offset, so a prefill is ceil(n / maxBatch) submits.
const maxBatch = 32

// maxPlanes is the fused Mat table: limb 0 plus seven further digit planes.
const maxPlanes = 8

// chunkFor is how many tokens one command buffer may run.
// DXGI_ERROR_DEVICE_HUNG is Windows ending a submit that runs past the
// timeout, about two seconds. A 30-layer chart with one plane on each of
// its four fused linears fits 32 tokens in that window. A chart whose
// linears carry more planes, which is hodge-5.1, does not: the same 32
// tokens remove the device during prefill.
func chunkFor(cost int) int {
	const fit = 30 * 4
	if cost <= fit {
		return maxBatch
	}
	n := fit * maxBatch * 3 / 4 / cost
	if n < 1 {
		return 1
	}
	if n > maxBatch {
		return maxBatch
	}
	return n
}

// gpuRT is the resident device. Every weight is uploaded once. A token is
// one recorded command buffer, not a fresh upload of each matrix.
type gpuRT struct {
	dev               *rt.Dev
	info              uint32
	stride            int // bytes between step records: the uniform offset alignment
	ctx               int
	stepID            uint32
	logits            uint32
	xBuf, yBuf, wrBuf uint32
	embedP            uint32
	embedX            uint32
	chunks            []embChunk
	ropeP, attnP      uint32
	ropeX, attnX      uint32
	addP, addG        uint32
	addX              uint32
	normD, normF      uint32
	normFin           uint32
	subP              uint32 // attention sub-norm; normD when H*HD equals hidden
	fin               uint32
	layers            []gpuLayer
	hl                map[string]uint32
	fused             int
	screen            *lmScreen
	batch             *batchRT
	shelf             *shelfRT
	// kv is the token at each position the cache holds, valid as a prefix:
	// row p was computed after exactly kv[:p+1]. last and lastPos are the
	// dialog whose last token is in x.
	kv      []int
	last    *dialog
	lastPos int
	dotSrc  string // the dot4 the kernels were built with
	// tok is one token's layer ops, recorded once. The ops at the indices in dyn
	// read the step record, so each token sets their dynamic offset.
	tok  []rt.Op
	dyn  []int
	head []rt.Op
	// stepN's scratch, kept between tokens.
	ops  []rt.Op
	recs []byte
	// cost is the digit-plane passes in one token. chunkFor uses it to keep
	// a prefill submit inside the Windows device timeout.
	cost int
}

type gpuLayer struct {
	attnN, ffnN, sub, ffnS uint32
	rope, attn             uint32
	qkv, o                 []rt.Op
	gateUp, down           []rt.Op
	pre                    []rt.Op // attention through the ffn norm
	// For the batched prefill: the fused linears (nil when a linear is not
	// fused) and the norm weights' buffers.
	fq, fo, fgu, fd     *fusedLin
	wAn, wFn, wSub, wFs string
}

// fusedLin is one hlx dispatch's weights and shape, which the batched
// prefill's hlxb dispatch reuses.
type fusedLin struct {
	wn, mn      string
	k, nm, maxl int
	resid       bool
	dests       []string
	x, y        uint32
}

// embChunk is one slice of the tied embedding. embed.wgsl reads st.row inside
// this chunk, and lmhead.wgsl writes this chunk's rows of logits.
type embChunk struct {
	start  int
	rows   int
	buf    uint32 // the device buffer, for reading rows back (hesrun.go keys)
	embedG uint32
	lmP    uint32
	lmG    uint32
	lmX    uint32
	lmY    uint32
}

type linG struct {
	pipe []uint32
	grp  []uint32
	x    uint32
}

func (m *Model) boot() error {
	if m.qnAny() {
		return fmt.Errorf("%s: q/k norm is not in this Dawn step", m.name)
	}
	dev, err := rt.Open()
	if err != nil {
		return err
	}
	if err := m.bootOn(dev); err != nil {
		dev.Close()
		return err
	}
	return nil
}

func (m *Model) bootOn(dev *rt.Dev) error {
	info, align, err := dev.Info()
	if err != nil {
		return fmt.Errorf("%s is older than this ModelRun (%v); rebuild it with dawnrt\\build.bat", rt.Exe(), err)
	}
	if align < 16 {
		align = 256
	}
	g := &gpuRT{dev: dev, info: info, stride: align, ctx: 4096, hl: map[string]uint32{}}
	k := paths.Default().KernelsDir()
	read := func(name string) (string, error) {
		b, err := os.ReadFile(filepath.Join(k, name))
		return string(b), err
	}
	soft, err := read("dot4_soft.wgsl")
	if err != nil {
		return err
	}
	lin, err := read("hodgelinear.wgsl")
	if err != nil {
		return err
	}
	hlx, err := ownKernel("hlx.wgsl")
	if err != nil {
		return err
	}
	add, err := ownKernel("add.wgsl")
	if err != nil {
		return err
	}
	// The hardware dot4 when the device has it. It is the same integer sum.
	dot, dotName := soft, "soft"
	if info&rt.InfoDot4 != 0 && os.Getenv("MODELRUN_DOT4") != "soft" {
		if native, err := read("dot4_native.wgsl"); err == nil && dev.Shader("hlx", native+"\n"+hlx) == nil {
			dot, dotName = native, "native"
		}
	}
	if dotName == "soft" {
		if err = dev.Shader("hlx", soft+"\n"+hlx); err != nil {
			return err
		}
	}
	g.dotSrc = dot
	if err = dev.Shader("hl", dot+"\n"+lin); err != nil {
		return err
	}
	if err = dev.Shader("add", add); err != nil {
		return err
	}
	head, err := ownKernel("lmhead.wgsl")
	if err != nil {
		return err
	}
	if err = dev.Shader("lmhead.wgsl", head); err != nil {
		return err
	}
	logf("device  dot4 %s  fast %v  uniform align %d", dotName, info&rt.InfoFast != 0, align)
	names := []string{"embed.wgsl", "norm.wgsl", "rope.wgsl", "attention.wgsl"}
	if m.floatAny() {
		names = append(names, "dense.wgsl")
	}
	for _, n := range names {
		s, err := read(n)
		if err != nil {
			return err
		}
		if err = dev.Shader(n, s); err != nil {
			return err
		}
	}
	if err = g.alloc(m); err != nil {
		return err
	}
	m.gpu = g
	m.embed = nil
	logf("weights resident  %d layers  ctx %d", m.L, g.ctx)
	return nil
}

func (m *Model) floatAny() bool {
	for i := range m.layers {
		L := &m.layers[i]
		if L.q.f != nil || L.k.f != nil || L.v.f != nil || L.o.f != nil || L.gate.f != nil || L.up.f != nil || L.down.f != nil {
			return true
		}
	}
	return false
}

func (m *Model) qnAny() bool {
	for i := range m.layers {
		if m.layers[i].qn != nil || m.layers[i].kn != nil {
			return true
		}
	}
	return false
}

func (g *gpuRT) alloc(m *Model) error {
	d := g.dev
	maxK := m.D
	if m.F > maxK {
		maxK = m.F
	}
	A := m.H * m.HD
	if A > maxK {
		maxK = A
	}
	for _, b := range []struct {
		name string
		n    int
	}{
		{"x", m.D * 4}, {"ao", A * 4}, {"qf", A * 4},
		{"kf", m.KV * m.HD * 4}, {"vf", m.KV * m.HD * 4},
		{"gv", m.F * 4}, {"uv", m.F * 4}, {"wr", m.D * 4},
		{"y", maxK * 4}, {"xq", maxK}, {"xs", 16},
		{"att", m.H * g.ctx * 4},
		{"dummy1", 16}, {"dummy2", 16},
	} {
		id, err := d.Buf(b.name, nil, b.n)
		if err != nil {
			return err
		}
		switch b.name {
		case "x":
			g.xBuf = id
		case "y":
			g.yBuf = id
		case "wr":
			g.wrBuf = id
		}
	}
	per := g.ctx * m.KV * m.HD * 4
	if _, err := d.Buf("kc", nil, m.L*per); err != nil {
		return err
	}
	if _, err := d.Buf("vc", nil, m.L*per); err != nil {
		return err
	}
	var err error
	if g.stepID, err = d.Buf("step", nil, maxBatch*g.stride); err != nil {
		return err
	}
	if g.logits, err = d.Buf("logits", nil, m.Vocab*4); err != nil {
		return err
	}
	ones := make([]float32, maxK)
	for i := range ones {
		ones[i] = 1
	}
	if _, err = d.Buf("ones", f32b(ones), len(ones)*4); err != nil {
		return err
	}
	if m.floatAny() {
		if _, err = d.Buf("zbias", []byte{0, 0, 0, 0}, 4); err != nil {
			return err
		}
	}
	logf("uploading embedding  %d bytes", len(m.embed))
	if err = g.uploadEmbed(m); err != nil {
		return err
	}
	if _, err = d.Buf("final", f32b(m.final), len(m.final)*4); err != nil {
		return err
	}

	scale := "1"
	if m.embedScale {
		scale = strconv.FormatFloat(math.Sqrt(float64(m.D)), 'f', -1, 64)
	}
	// The step record is a dynamic uniform: token i of a submit reads record i.
	uni := []byte{rt.KindUniDyn, rt.KindRO, rt.KindRW}
	if g.embedP, err = d.Pipe("embed", "embed.wgsl", "main", "D="+strconv.Itoa(m.D)+" SCALE="+scale, uni); err != nil {
		return err
	}
	for i := range g.chunks {
		c := &g.chunks[i]
		name := fmt.Sprintf("emb%d", c.start)
		if c.embedG, err = d.Group(name, "embed", []rt.Bind{{Buf: "step", Size: 16}, {Buf: name}, {Buf: "x"}}); err != nil {
			return err
		}
	}
	g.embedX = uint32((m.D + 255) / 256)

	act, err := actID(m.act)
	if err != nil {
		return err
	}
	nk := []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRW}
	eps := strconv.FormatFloat(m.eps, 'g', -1, 64)
	gem := "0"
	if m.gemma {
		gem = "1"
	}
	base := "K=%d EPS=" + eps + " GEMMA=" + gem
	if g.normD, err = d.Pipe("normD", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=0 ACT=0", m.D), nk); err != nil {
		return err
	}
	if g.normF, err = d.Pipe("normF", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=1 ACT=%d", m.F, act), nk); err != nil {
		return err
	}
	if g.normFin, err = d.Pipe("normFin", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=2 ACT=0", m.D), nk); err != nil {
		return err
	}
	// The attention sub-norm covers ao, which is H*HD wide. normD is compiled
	// at the hidden size. Sharing it when those differ reads past ao.
	g.subP = g.normD
	subPipe := "normD"
	if A != m.D {
		if A%4 != 0 {
			return fmt.Errorf("attention width %d is not a multiple of 4", A)
		}
		if g.subP, err = d.Pipe("normA", "norm.wgsl", "main", fmt.Sprintf(base+" MODE=0 ACT=0", A), nk); err != nil {
			return err
		}
		subPipe = "normA"
	}
	ropeOv := fmt.Sprintf("H=%d KVH=%d HD=%d THETA=%s SCALING=%d FACTOR=%s LOW_FF=%s HIGH_FF=%s ORIG=%s",
		m.H, m.KV, m.HD, strconv.FormatFloat(m.theta, 'g', -1, 64), m.ropeScale,
		strconv.FormatFloat(m.ropeFactor, 'g', -1, 64),
		strconv.FormatFloat(m.ropeLow, 'g', -1, 64),
		strconv.FormatFloat(m.ropeHigh, 'g', -1, 64),
		strconv.FormatFloat(m.ropeOrig, 'g', -1, 64))
	rk := []byte{rt.KindUniDyn, rt.KindRW, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW}
	if g.ropeP, err = d.Pipe("rope", "rope.wgsl", "main", ropeOv, rk); err != nil {
		return err
	}
	g.ropeX = uint32((m.H*m.HD/2 + m.KV*m.HD/2 + 63) / 64)
	denom := "0"
	if m.queryScale > 0 {
		denom = strconv.FormatFloat(m.queryScale, 'g', -1, 64)
	}
	ak := []byte{rt.KindUniDyn, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW}
	// attng.wgsl reads each cached row once per grouped-query group; attn.wgsl
	// once per head. Both are attention.wgsl's arithmetic. attention.wgsl is
	// the fallback for a head width the vec4 loads cannot take.
	src, x := "attention.wgsl", uint32(m.H)
	switch {
	case attnGrouped(m):
		src, x = "attng.wgsl", uint32(m.KV)
	case m.HD%4 == 0:
		src = "attn.wgsl"
	}
	if src != "attention.wgsl" {
		k, err := ownKernel(src)
		if err != nil {
			return err
		}
		if err := d.Shader(src, k); err != nil {
			return err
		}
	}
	if g.attnP, err = d.Pipe("attn", src, "main", fmt.Sprintf("H=%d KVH=%d HD=%d CTX=%d DENOM=%s", m.H, m.KV, m.HD, g.ctx, denom), ak); err != nil {
		return err
	}
	g.attnX = x
	if g.addP, err = d.Pipe("addD", "add", "main", fmt.Sprintf("N=%d", m.D), []byte{rt.KindRW, rt.KindRO}); err != nil {
		return err
	}
	if g.addG, err = d.Group("addD", "addD", []rt.Bind{{Buf: "x"}, {Buf: "wr"}}); err != nil {
		return err
	}
	g.addX = uint32((m.D + 255) / 256)
	lk := []byte{rt.KindRO, rt.KindRO, rt.KindRW}
	for i := range g.chunks {
		c := &g.chunks[i]
		name := fmt.Sprintf("lm%d", c.start)
		if c.lmP, err = d.Pipe(name, "lmhead.wgsl", "main", fmt.Sprintf("D=%d ROWS=%d", m.D, c.rows), lk); err != nil {
			return err
		}
		if c.lmG, err = d.Group(name, name, []rt.Bind{
			{Buf: "y"},
			{Buf: fmt.Sprintf("emb%d", c.start)},
			{Buf: "logits", Off: c.start * 4, Size: c.rows * 4},
		}); err != nil {
			return err
		}
		c.lmX, c.lmY = grid(uint32((c.rows + 63) / 64)) // 64 rows per workgroup
	}
	if g.fin, err = g.normGroup("final", "normFin", "x", "x", m.final, m.D); err != nil {
		return err
	}

	g.layers = make([]gpuLayer, m.L)
	for i := range m.layers {
		L := &m.layers[i]
		gl := &g.layers[i]
		if gl.attnN, err = g.normGroup(fmt.Sprintf("L%d-an", i), "normD", "x", "x", L.attnN, m.D); err != nil {
			return err
		}
		if gl.ffnN, err = g.normGroup(fmt.Sprintf("L%d-fn", i), "normD", "x", "x", L.ffnN, m.D); err != nil {
			return err
		}
		if gl.sub, err = g.normGroup(fmt.Sprintf("L%d-sub", i), subPipe, "ao", "ao", orOnes(L.attnS, A), orLen(L.attnS, A)); err != nil {
			return err
		}
		if gl.ffnS, err = g.normGroup(fmt.Sprintf("L%d-fs", i), "normF", "gv", "uv", orOnes(L.ffnS, m.F), orLen(L.ffnS, m.F)); err != nil {
			return err
		}
		gl.wAn = normWeight(fmt.Sprintf("L%d-an", i), L.attnN)
		gl.wFn = normWeight(fmt.Sprintf("L%d-fn", i), L.ffnN)
		gl.wSub = normWeight(fmt.Sprintf("L%d-sub", i), L.attnS)
		gl.wFs = normWeight(fmt.Sprintf("L%d-fs", i), L.ffnS)
		if gl.qkv, gl.fq, err = g.project(fmt.Sprintf("L%d-qkv", i), []lin{L.q, L.k, L.v}, []string{"qf", "kf", "vf"}); err != nil {
			return err
		}
		if gl.o, gl.fo, err = g.residual(fmt.Sprintf("L%d-o", i), L.o); err != nil {
			return err
		}
		if L.gate.w != nil || L.gate.f != nil {
			if gl.gateUp, gl.fgu, err = g.project(fmt.Sprintf("L%d-gu", i), []lin{L.gate, L.up}, []string{"gv", "uv"}); err != nil {
				return err
			}
			if gl.down, gl.fd, err = g.residual(fmt.Sprintf("L%d-d", i), L.down); err != nil {
				return err
			}
		}
		off := i * per
		if gl.rope, err = d.Group(fmt.Sprintf("L%d-rope", i), "rope", []rt.Bind{
			{Buf: "step", Size: 16}, {Buf: "qf"}, {Buf: "kf"}, {Buf: "vf"},
			{Buf: "kc", Off: off, Size: per}, {Buf: "vc", Off: off, Size: per},
		}); err != nil {
			return err
		}
		if gl.attn, err = d.Group(fmt.Sprintf("L%d-attn", i), "attn", []rt.Bind{
			{Buf: "step", Size: 16}, {Buf: "qf"}, {Buf: "kc", Off: off, Size: per}, {Buf: "vc", Off: off, Size: per},
			{Buf: "att"}, {Buf: "ao"},
		}); err != nil {
			return err
		}
		g.moePre(i)
		L.q, L.k, L.v, L.o = lin{}, lin{}, lin{}, lin{}
		L.gate, L.up, L.down = lin{}, lin{}, lin{}
		L.attnN, L.ffnN, L.attnS, L.ffnS = nil, nil, nil, nil
		logf("layer %d resident", i)
	}
	m.final = nil
	if err := g.bootScreen(m); err != nil {
		return err
	}
	if err := g.bootBatch(m); err != nil {
		return err
	}
	if err := g.bootShelf(m); err != nil {
		return err
	}
	g.record()
	logf("fused linears %d  dispatches per token %d", g.fused, len(g.tok)+1)
	if n := chunkFor(g.cost); n < maxBatch {
		logf("prefill submit %d tokens", n)
	}
	return nil
}

// record builds one token's ops once. Embedding, rope and attention read the
// step record, so they are the ops whose dynamic offset changes per token.
func (g *gpuRT) record() {
	var ops []rt.Op
	var dyn []int
	add := func(o ...rt.Op) { ops = append(ops, o...) }
	addDyn := func(pipe, group, x uint32) {
		dyn = append(dyn, len(ops))
		ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: pipe, Group: group, X: x, Y: 1, Z: 1})
	}
	for li := range g.layers {
		L := &g.layers[li]
		add(disp(g.normD, L.attnN, 1)...)
		add(L.qkv...)
		addDyn(g.ropeP, L.rope, g.ropeX)
		addDyn(g.attnP, L.attn, g.attnX)
		add(disp(g.subP, L.sub, 1)...)
		add(L.o...)
		add(disp(g.normD, L.ffnN, 1)...)
		add(L.gateUp...)
		add(disp(g.normF, L.ffnS, 1)...)
		add(L.down...)
	}
	g.tok, g.dyn = ops, dyn
	g.head = disp(g.normFin, g.fin, 1)
	for i := range g.chunks {
		c := &g.chunks[i]
		g.head = append(g.head, rt.Op{Tag: rt.OpDispatch, Pipe: c.lmP, Group: c.lmG, X: c.lmX, Y: c.lmY, Z: 1})
	}
}

// packedFusable is a set of packed linears hlx.wgsl runs in one dispatch:
// at most three, one input width, at most seven limbs past the first.
func packedFusable(ls []lin) bool {
	if len(ls) == 0 || len(ls) > 3 || os.Getenv("MODELRUN_FUSE") == "0" {
		return false
	}
	for _, l := range ls {
		if l.f != nil || l.w == nil || l.k != ls[0].k || l.k%4 != 0 || l.n%4 != 0 || len(l.limbs) > maxPlanes-1 {
			return false
		}
	}
	return true
}

// project is ls, which share one input, into dests. Packed, it is one hlx dispatch.
func (g *gpuRT) project(name string, ls []lin, dests []string) ([]rt.Op, *fusedLin, error) {
	if packedFusable(ls) {
		return g.hlxOps(name, ls, dests, false)
	}
	var ops []rt.Op
	for i, l := range ls {
		lg, err := g.linGroups(fmt.Sprintf("%s%d", name, i), dests[i], l)
		if err != nil {
			return nil, nil, err
		}
		ops = append(ops, linOps(lg)...)
	}
	return ops, nil, nil
}

// residual is x += l(input). Packed, the add is inside the hlx dispatch.
func (g *gpuRT) residual(name string, l lin) ([]rt.Op, *fusedLin, error) {
	if packedFusable([]lin{l}) {
		return g.hlxOps(name, []lin{l}, []string{"x"}, true)
	}
	lg, err := g.linGroups(name, "wr", l)
	if err != nil {
		return nil, nil, err
	}
	return append(linOps(lg), disp(g.addP, g.addG, g.addX)...), nil, nil
}

// hlxOps uploads every limb of ls into one buffer, with the table hlx.wgsl
// reads, and returns the one dispatch.
func (g *gpuRT) hlxOps(name string, ls []lin, dests []string, resid bool) ([]rt.Op, *fusedLin, error) {
	le := binary.LittleEndian
	size := 0
	for _, l := range ls {
		size += len(l.w)
		for _, lb := range l.limbs {
			size += len(lb.w)
		}
	}
	blob := make([]byte, 0, size)
	var meta []byte
	start, maxl := 0, 1
	for _, l := range ls {
		np := l.n / 4
		ws := [][]byte{l.w}
		scale := []float32{l.s}
		center := []int32{1}
		for _, lb := range l.limbs {
			ws = append(ws, lb.w)
			scale = append(scale, lb.s)
			center = append(center, int32(lb.center))
		}
		maxl = max(maxl, len(ws))
		var woff [maxPlanes]uint32
		if len(ws) > len(woff) {
			return nil, nil, fmt.Errorf("%s: %d planes, the fused table holds %d", name, len(ws), len(woff))
		}
		for i, w := range ws {
			if len(w) != np*l.k {
				return nil, nil, fmt.Errorf("%s: limb %d is %d bytes, want %d", name, i, len(w), np*l.k)
			}
			woff[i] = uint32(len(blob) / 4)
			blob = append(blob, w...)
		}
		meta = le.AppendUint32(meta, uint32(start))
		meta = le.AppendUint32(meta, uint32(np))
		meta = le.AppendUint32(meta, uint32(len(ws)))
		meta = le.AppendUint32(meta, 0)
		for i := range woff {
			meta = le.AppendUint32(meta, woff[i])
		}
		for i := range woff {
			var c int32
			if i < len(center) {
				c = center[i]
			}
			meta = le.AppendUint32(meta, uint32(c))
		}
		for i := range woff {
			var v float32
			if i < len(scale) {
				v = scale[i]
			}
			meta = le.AppendUint32(meta, math.Float32bits(v))
		}
		start += np
	}
	wn, mn := name+"-w", name+"-m"
	if _, err := g.dev.Buf(wn, blob, len(blob)); err != nil {
		return nil, nil, err
	}
	if _, err := g.dev.Buf(mn, meta, len(meta)); err != nil {
		return nil, nil, err
	}
	key := fmt.Sprintf("K=%d NM=%d MAXL=%d RESID=%v", ls[0].k, len(ls), maxl, resid)
	pipe := "hlx-" + key
	id, ok := g.hl[pipe]
	if !ok {
		var err error
		kinds := []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRW}
		if id, err = g.dev.Pipe(pipe, "hlx", "main", key, kinds); err != nil {
			return nil, nil, err
		}
		g.hl[pipe] = id
	}
	// Unused outputs bind two distinct dummies: two writable bindings of one buffer alias.
	out := [3]string{"dummy1", "dummy2", "dummy2"}
	switch len(dests) {
	case 1:
		out = [3]string{dests[0], "dummy1", "dummy2"}
	case 2:
		out = [3]string{dests[0], dests[1], "dummy1"}
	case 3:
		out = [3]string{dests[0], dests[1], dests[2]}
	}
	gid, err := g.dev.Group(name, pipe, []rt.Bind{
		{Buf: "xq"}, {Buf: "xs", Size: 16}, {Buf: wn}, {Buf: mn},
		{Buf: out[0]}, {Buf: out[1]}, {Buf: out[2]},
	})
	if err != nil {
		return nil, nil, err
	}
	g.fused++
	g.cost += maxl
	groups := uint32((start + 7) / 8)
	x, y := groups, uint32(1)
	for x > 65535 {
		y *= 2
		x = (groups + y - 1) / y
	}
	f := &fusedLin{wn: wn, mn: mn, k: ls[0].k, nm: len(ls), maxl: maxl, resid: resid, dests: dests, x: x, y: y}
	return []rt.Op{{Tag: rt.OpDispatch, Pipe: id, Group: gid, X: x, Y: y, Z: 1}}, f, nil
}

// attnGrouped is a shape attng.wgsl holds: at most 8 query heads per group
// and the group's q (G * HD floats) in 2048.
func attnGrouped(m *Model) bool {
	if m.KV == 0 || m.H%m.KV != 0 || m.HD%4 != 0 || os.Getenv("MODELRUN_ATTN") == "head" {
		return false
	}
	gq := m.H / m.KV
	return gq <= 8 && gq*m.HD <= 2048
}

// normWeight is the buffer normGroup binds for w: its own, or "ones".
func normWeight(name string, w []float32) string {
	if w == nil {
		return "ones"
	}
	return name + "-w"
}

func orOnes(w []float32, n int) []float32 {
	if w != nil {
		return w
	}
	return nil
}

func orLen(w []float32, n int) int {
	if w != nil {
		return len(w)
	}
	return n
}

func (g *gpuRT) normGroup(name, pipe, a, b string, w []float32, n int) (uint32, error) {
	buf := "ones"
	if w != nil {
		buf = name + "-w"
		if _, err := g.dev.Buf(buf, f32b(w), n*4); err != nil {
			return 0, err
		}
	}
	return g.dev.Group(name, pipe, []rt.Bind{
		{Buf: a}, {Buf: b}, {Buf: buf, Size: n * 4}, {Buf: "xq"}, {Buf: "xs", Size: 16}, {Buf: "y"},
	})
}

// linGroups is one linear as hodgelinear.wgsl dispatches, one per limb, or dense.wgsl.
func (g *gpuRT) linGroups(name, dest string, l lin) (linG, error) {
	if l.f != nil {
		return g.denseGroups(name, dest, l)
	}
	var out linG
	out.x = uint32((l.n/4 + 7) / 8)
	type part struct {
		w     []byte
		s     float32
		c     int
		accum bool
	}
	parts := []part{{l.w, l.s, 1, false}}
	for _, lb := range l.limbs {
		parts = append(parts, part{lb.w, lb.s, lb.center, true})
	}
	for i, p := range parts {
		wn := fmt.Sprintf("%s-w%d", name, i)
		sn := fmt.Sprintf("%s-s%d", name, i)
		if _, err := g.dev.Buf(wn, p.w, len(p.w)); err != nil {
			return linG{}, err
		}
		if _, err := g.dev.Buf(sn, f32b([]float32{p.s}), 16); err != nil {
			return linG{}, err
		}
		key := fmt.Sprintf("K=%d NP=%d ACCUM=%v CENTER=%d", l.k, l.n/4, p.accum, p.c)
		id, ok := g.hl[key]
		if !ok {
			var err error
			id, err = g.dev.Pipe("hl-"+key, "hl", "main", key, []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW})
			if err != nil {
				return linG{}, err
			}
			g.hl[key] = id
		}
		gid, err := g.dev.Group(fmt.Sprintf("%s-g%d", name, i), "hl-"+key, []rt.Bind{
			{Buf: "xq"}, {Buf: "xs", Size: 16}, {Buf: wn}, {Buf: sn, Size: 16}, {Buf: dest},
		})
		if err != nil {
			return linG{}, err
		}
		out.pipe = append(out.pipe, id)
		out.grp = append(out.grp, gid)
	}
	g.cost += len(parts)
	return out, nil
}

func (g *gpuRT) denseGroups(name, dest string, l lin) (linG, error) {
	if l.k%2 != 0 || len(l.f) != l.n*l.k*2 {
		return linG{}, fmt.Errorf("%s: dense weight %d bytes", name, len(l.f))
	}
	wn := name + "-w"
	if _, err := g.dev.Buf(wn, l.f, len(l.f)); err != nil {
		return linG{}, err
	}
	biasName := "zbias"
	bias := 0
	bb := l.bias
	if len(bb) > 0 {
		bias = 1
		biasName = name + "-b"
		if len(bb)%4 != 0 {
			bb = append(append([]byte{}, bb...), 0, 0)
		}
		if _, err := g.dev.Buf(biasName, bb, len(bb)); err != nil {
			return linG{}, err
		}
	}
	key := fmt.Sprintf("N=%d K=%d BIAS=%d", l.n, l.k, bias)
	id, ok := g.hl[key]
	if !ok {
		var err error
		id, err = g.dev.Pipe("dense-"+key, "dense.wgsl", "main", key, []byte{rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRO})
		if err != nil {
			return linG{}, err
		}
		g.hl[key] = id
	}
	gid, err := g.dev.Group(name, "dense-"+key, []rt.Bind{
		{Buf: "y"}, {Buf: wn}, {Buf: dest}, {Buf: biasName},
	})
	if err != nil {
		return linG{}, err
	}
	g.cost += 1
	return linG{pipe: []uint32{id}, grp: []uint32{gid}, x: uint32((l.n + 7) / 8)}, nil
}

// step runs ids through the layers. With req, the last submit also runs the
// head and d.lg is the sampler's logits; without it d.lg is nil.
func (m *Model) step(d *dialog, ids []int, req *headReq) error {
	if m.gpu == nil {
		return fmt.Errorf("no device")
	}
	d.lg = nil
	if err := m.restore(d); err != nil {
		return err
	}
	limit := chunkFor(m.gpu.cost)
	for off := 0; off < len(ids); off += limit {
		if interrupted.Load() {
			return ErrInterrupted
		}
		n := min(len(ids)-off, limit)
		var r *headReq
		if off+n == len(ids) {
			r = req
		}
		if err := m.runChunk(d, ids[off:off+n], r); err != nil {
			return err
		}
	}
	return nil
}

// sync waits until the device has run every submit.
func (m *Model) sync() error {
	if m.gpu == nil {
		return fmt.Errorf("no device")
	}
	_, err := m.gpu.dev.Run2(nil, nil, nil, true)
	return err
}

// restore makes the cache hold d's tokens: whatever another dialog wrote over
// since d last ran is run again.
func (m *Model) restore(d *dialog) error {
	g := m.gpu
	c := commonPrefix(g.kv, d.ids[:d.pos])
	if c == d.pos {
		return nil
	}
	if err := m.fit(d.ids[:d.pos]); err != nil {
		return err
	}
	if c = commonPrefix(g.kv, d.ids[:d.pos]); c == d.pos {
		return nil
	}
	again := append([]int(nil), d.ids[c:d.pos]...)
	logf("replaying %d tokens another chat wrote over", len(again))
	d.ids, d.pos = d.ids[:c], c
	limit := chunkFor(g.cost)
	for off := 0; off < len(again); off += limit {
		n := min(len(again)-off, limit)
		if err := m.runChunk(d, again[off:off+n], nil); err != nil {
			return err
		}
	}
	return nil
}

func (m *Model) runChunk(d *dialog, ids []int, req *headReq) error {
	if len(m.slots) > 0 {
		for _, id := range ids {
			if !m.isSlot(id) {
				continue
			}
			// Image rows and phases are written only by the batched prefill.
			if m.gpu.batch == nil || m.mixed || m.moe {
				return fmt.Errorf("images need the batched prefill, which this model does not run")
			}
			return m.stepB(d, ids, req)
		}
	}
	if m.mixed {
		return m.stepMix(d, ids, req)
	}
	if m.moe {
		return m.stepMoE(d, ids, req)
	}
	if m.gpu.batch != nil && len(ids) > 1 {
		return m.stepB(d, ids, req)
	}
	return m.stepN(d, ids, req)
}

// ran records ids at d's position in the dialog and in the cache.
func (g *gpuRT) ran(d *dialog, ids []int) {
	g.shelf.noteWrite(d.pos + len(ids))
	if len(g.kv) > d.pos {
		g.kv = g.kv[:d.pos]
	}
	g.kv = append(g.kv, ids...)
	d.ids = append(d.ids[:d.pos], ids...)
	d.pos += len(ids)
	g.last, g.lastPos = d, d.pos
}

// head runs only the head, on the hidden state of d's last token.
func (m *Model) head(d *dialog, req *headReq) error {
	g := m.gpu
	if g == nil {
		return fmt.Errorf("no device")
	}
	if d.pos == 0 {
		return fmt.Errorf("head: no token")
	}
	if g.last != d || g.lastPos != d.pos {
		// x is another dialog's: run d's last token again, with the head.
		last := d.ids[d.pos-1]
		d.ids, d.pos = d.ids[:d.pos-1], d.pos-1
		return m.step(d, []int{last}, req)
	}
	return m.runHead(d, nil, nil, req)
}

// stepN runs ids (at most maxBatch) through every layer in one submit. Token i
// reads step record i, at byte offset i*stride, for its position and its
// embedding row. Only the last token's logits are computed, and only with req.
// A submit without the head does not wait: the next one queues behind it.
func (m *Model) stepN(d *dialog, ids []int, req *headReq) error {
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
	need := len(ids) * g.stride
	if cap(g.recs) < need {
		g.recs = make([]byte, need)
	}
	recs := g.recs[:need]
	clear(recs)
	ops := g.ops[:0]
	for i, id := range ids {
		ch, row, err := g.chunk(id)
		if err != nil {
			return err
		}
		at := i * g.stride
		binary.LittleEndian.PutUint32(recs[at:], uint32(d.pos+i))
		binary.LittleEndian.PutUint32(recs[at+4:], uint32(row))
		dyn := uint32(at)
		ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: g.embedP, Group: ch.embedG, X: g.embedX, Y: 1, Z: 1, Dyn: dyn})
		base := len(ops)
		ops = append(ops, g.tok...)
		for _, j := range g.dyn {
			ops[base+j].Dyn = dyn
		}
	}
	g.ops = ops
	writes := []rt.Write{{Buf: g.stepID, Data: recs}}
	g.ran(d, ids)
	if req != nil {
		return m.runHead(d, writes, ops, req)
	}
	_, err := g.dev.Run2(writes, ops, nil, false)
	return err
}

// Dawn's storage binding limit. A larger embedding is several chunks.
const maxStorage = 2147483648

func chunkRows(rowBytes, vocab int) int {
	if rowBytes <= 0 || vocab <= 0 {
		return vocab
	}
	if vocab*rowBytes <= maxStorage {
		return vocab
	}
	n := maxStorage / rowBytes
	n -= n % 64
	if n < 64 {
		n = 64
	}
	return n
}

func (g *gpuRT) uploadEmbed(m *Model) error {
	rowBytes := m.D * 2
	if rowBytes <= 0 || len(m.embed) < m.Vocab*rowBytes {
		return fmt.Errorf("embedding is %d bytes, want %d", len(m.embed), m.Vocab*rowBytes)
	}
	step := chunkRows(rowBytes, m.Vocab)
	for start := 0; start < m.Vocab; {
		n := step
		if start+n > m.Vocab {
			n = m.Vocab - start
		}
		blob := m.embed[start*rowBytes : (start+n)*rowBytes]
		name := fmt.Sprintf("emb%d", start)
		id, err := g.dev.Buf(name, blob, len(blob))
		if err != nil {
			return err
		}
		g.chunks = append(g.chunks, embChunk{start: start, rows: n, buf: id})
		start += n
	}
	return nil
}

func (g *gpuRT) chunk(id int) (*embChunk, int, error) {
	for i := range g.chunks {
		c := &g.chunks[i]
		if id >= c.start && id < c.start+c.rows {
			return c, id - c.start, nil
		}
	}
	return nil, 0, fmt.Errorf("token %d is outside the embedding", id)
}

func disp(pipe, group, x uint32) []rt.Op {
	return []rt.Op{{Tag: rt.OpDispatch, Pipe: pipe, Group: group, X: x, Y: 1, Z: 1}}
}

func linOps(l linG) []rt.Op {
	ops := make([]rt.Op, len(l.pipe))
	for i := range l.pipe {
		ops[i] = rt.Op{Tag: rt.OpDispatch, Pipe: l.pipe[i], Group: l.grp[i], X: l.x, Y: 1, Z: 1}
	}
	return ops
}

func f32b(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}
