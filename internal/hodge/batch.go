package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strconv"

	"gptrun/internal/rt"
)

// The batched prefill runs up to maxBatch prompt tokens through each layer
// together: one norm, one fused linear, one rope and one attention dispatch per
// layer step instead of one per token, and every weight word is read once per
// 8 tokens (hlxb.wgsl) instead of once per token. Each batched kernel does, for
// each token, the arithmetic its one-token kernel does, so the cache and the
// hidden state are the bits a token-by-token prefill leaves. It needs every
// linear fused and at most two embedding chunks; otherwise prefill runs a
// token at a time. MODELRUN_BATCH=0 turns it off.

type batchRT struct {
	embP, embG          uint32
	embX                uint32
	normD, normF, normA uint32
	ropeP, attnP        uint32
	ropeX, attnX        uint32
	bstep, bn, bids, bx uint32
	bslot, bph          uint32 // image rows [token][D] and phases [token] for this batch
	layers              []batchLayer
	xBytes              int
	ops                 []rt.Op
	ids                 []byte
	slot, ph            []byte
}

type batchLayer struct {
	an, sub, fn, fs  uint32
	rope, attn       uint32
	qkv, o, gu, down rt.Op
}

func (g *gpuRT) bootBatch(m *Model) error {
	// The batch path has no chart. A mixture stays on stepMoE.
	if m.moe || m.mixed || os.Getenv("MODELRUN_BATCH") == "0" || len(g.chunks) > 2 {
		return nil
	}
	for i := range g.layers {
		L := &g.layers[i]
		if L.fq == nil || L.fo == nil || L.fgu == nil || L.fd == nil {
			return nil
		}
	}
	d := g.dev
	B := maxBatch
	A := m.H * m.HD
	maxK := max(m.D, m.F, A)
	srcs := map[string]string{}
	attnSrc, attnX := "attnb.wgsl", uint32(m.H)
	if attnGrouped(m) {
		attnSrc, attnX = "attngb.wgsl", uint32(m.KV)
	}
	for _, n := range []string{"hlxb.wgsl", "normb.wgsl", "ropeb.wgsl", attnSrc, "embedb.wgsl"} {
		s, err := ownKernel(n)
		if err != nil {
			return err
		}
		srcs[n] = s
	}
	if err := d.Shader("hlxb", g.dotSrc+"\n"+srcs["hlxb.wgsl"]); err != nil {
		return err
	}
	for _, n := range []string{"normb.wgsl", "ropeb.wgsl", attnSrc, "embedb.wgsl"} {
		if err := d.Shader(n, srcs[n]); err != nil {
			return err
		}
	}
	b := &batchRT{xBytes: m.D * 4}
	var err error
	for _, z := range []struct {
		name string
		n    int
	}{
		{"bx", B * m.D * 4}, {"bao", B * A * 4}, {"bqf", B * A * 4},
		{"bkf", B * m.KV * m.HD * 4}, {"bvf", B * m.KV * m.HD * 4},
		{"bgv", B * m.F * 4}, {"buv", B * m.F * 4},
		{"by", B * maxK * 4}, {"bxq", B * maxK}, {"bxs", B * 4},
		{"batt", B * m.H * g.ctx * 4},
	} {
		if _, err := d.Buf(z.name, nil, z.n); err != nil {
			return err
		}
	}
	if b.bstep, err = d.Buf("bstep", nil, 16); err != nil {
		return err
	}
	if b.bn, err = d.Buf("bn", nil, 16); err != nil {
		return err
	}
	if b.bids, err = d.Buf("bids", nil, B*4); err != nil {
		return err
	}
	if b.bx, err = d.Buf("bx", nil, 16); err != nil {
		return err
	}
	if b.bslot, err = d.Buf("bslot", nil, B*m.D*4); err != nil {
		return err
	}
	if b.bph, err = d.Buf("bph", nil, B*4); err != nil {
		return err
	}

	scale := "1"
	if m.embedScale {
		scale = strconv.FormatFloat(math.Sqrt(float64(m.D)), 'f', -1, 64)
	}
	r0 := g.chunks[0].rows
	if b.embP, err = d.Pipe("embedb", "embedb.wgsl", "main", fmt.Sprintf("D=%d R0=%d SCALE=%s SLOT=%d", m.D, r0, scale, m.Vocab),
		[]byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRO, rt.KindRO}); err != nil {
		return err
	}
	e1 := fmt.Sprintf("emb%d", g.chunks[0].start)
	if len(g.chunks) > 1 {
		e1 = fmt.Sprintf("emb%d", g.chunks[1].start)
	}
	if b.embG, err = d.Group("embedb", "embedb", []rt.Bind{
		{Buf: "bids"}, {Buf: fmt.Sprintf("emb%d", g.chunks[0].start)}, {Buf: e1}, {Buf: "bx"}, {Buf: "bn", Size: 16}, {Buf: "bslot"},
	}); err != nil {
		return err
	}
	b.embX = uint32((m.D + 255) / 256)

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
	if b.normD, err = d.Pipe("normbD", "normb.wgsl", "main", fmt.Sprintf(base+" MODE=0 ACT=0", m.D), nk); err != nil {
		return err
	}
	if b.normF, err = d.Pipe("normbF", "normb.wgsl", "main", fmt.Sprintf(base+" MODE=1 ACT=%d", m.F, act), nk); err != nil {
		return err
	}
	b.normA = b.normD
	subPipe := "normbD"
	if A != m.D {
		if b.normA, err = d.Pipe("normbA", "normb.wgsl", "main", fmt.Sprintf(base+" MODE=0 ACT=0", A), nk); err != nil {
			return err
		}
		subPipe = "normbA"
	}
	ropeOv := fmt.Sprintf("H=%d KVH=%d HD=%d THETA=%s SCALING=%d FACTOR=%s LOW_FF=%s HIGH_FF=%s ORIG=%s",
		m.H, m.KV, m.HD, strconv.FormatFloat(m.theta, 'g', -1, 64), m.ropeScale,
		strconv.FormatFloat(m.ropeFactor, 'g', -1, 64),
		strconv.FormatFloat(m.ropeLow, 'g', -1, 64),
		strconv.FormatFloat(m.ropeHigh, 'g', -1, 64),
		strconv.FormatFloat(m.ropeOrig, 'g', -1, 64))
	if b.ropeP, err = d.Pipe("ropeb", "ropeb.wgsl", "main", ropeOv,
		[]byte{rt.KindUni, rt.KindRW, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRO, rt.KindRO}); err != nil {
		return err
	}
	b.ropeX = g.ropeX
	denom := "0"
	if m.queryScale > 0 {
		denom = strconv.FormatFloat(m.queryScale, 'g', -1, 64)
	}
	b.attnX = attnX
	if b.attnP, err = d.Pipe("attnb", attnSrc, "main", fmt.Sprintf("H=%d KVH=%d HD=%d CTX=%d DENOM=%s", m.H, m.KV, m.HD, g.ctx, denom),
		[]byte{rt.KindUni, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRO}); err != nil {
		return err
	}

	normb := func(name, pipe, a, bb, w string, n int) (uint32, error) {
		return d.Group(name, pipe, []rt.Bind{
			{Buf: a}, {Buf: bb}, {Buf: w, Size: n * 4}, {Buf: "bxq"}, {Buf: "bxs"}, {Buf: "by"},
		})
	}
	per := g.ctx * m.KV * m.HD * 4
	b.layers = make([]batchLayer, len(g.layers))
	for i := range g.layers {
		L := &g.layers[i]
		bl := &b.layers[i]
		if bl.an, err = normb(fmt.Sprintf("B%d-an", i), "normbD", "bx", "bx", L.wAn, m.D); err != nil {
			return err
		}
		if bl.fn, err = normb(fmt.Sprintf("B%d-fn", i), "normbD", "bx", "bx", L.wFn, m.D); err != nil {
			return err
		}
		if bl.sub, err = normb(fmt.Sprintf("B%d-sub", i), subPipe, "bao", "bao", L.wSub, A); err != nil {
			return err
		}
		if bl.fs, err = normb(fmt.Sprintf("B%d-fs", i), "normbF", "bgv", "buv", L.wFs, m.F); err != nil {
			return err
		}
		for _, f := range []struct {
			name string
			fl   *fusedLin
			op   *rt.Op
		}{
			{fmt.Sprintf("B%d-qkv", i), L.fq, &bl.qkv}, {fmt.Sprintf("B%d-o", i), L.fo, &bl.o},
			{fmt.Sprintf("B%d-gu", i), L.fgu, &bl.gu}, {fmt.Sprintf("B%d-d", i), L.fd, &bl.down},
		} {
			if *f.op, err = g.hlxbOp(f.name, f.fl); err != nil {
				return err
			}
		}
		off := i * per
		if bl.rope, err = d.Group(fmt.Sprintf("B%d-rope", i), "ropeb", []rt.Bind{
			{Buf: "bstep", Size: 16}, {Buf: "bqf"}, {Buf: "bkf"}, {Buf: "bvf"},
			{Buf: "kc", Off: off, Size: per}, {Buf: "vc", Off: off, Size: per}, {Buf: "bn", Size: 16}, {Buf: "bph"},
		}); err != nil {
			return err
		}
		if bl.attn, err = d.Group(fmt.Sprintf("B%d-attn", i), "attnb", []rt.Bind{
			{Buf: "bstep", Size: 16}, {Buf: "bqf"}, {Buf: "kc", Off: off, Size: per}, {Buf: "vc", Off: off, Size: per},
			{Buf: "batt"}, {Buf: "bao"}, {Buf: "bn", Size: 16},
		}); err != nil {
			return err
		}
	}
	g.batch = b
	logf("batched prefill  %d tokens per submit", B)
	return nil
}

// batchDest is the batch buffer for a one-token buffer.
var batchDest = map[string]string{"qf": "bqf", "kf": "bkf", "vf": "bvf", "gv": "bgv", "uv": "buv", "x": "bx"}

func (g *gpuRT) hlxbOp(name string, f *fusedLin) (rt.Op, error) {
	key := fmt.Sprintf("K=%d NM=%d MAXL=%d RESID=%v", f.k, f.nm, f.maxl, f.resid)
	pipe := "hlxb-" + key
	id, ok := g.hl[pipe]
	if !ok {
		var err error
		kinds := []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRW}
		if id, err = g.dev.Pipe(pipe, "hlxb", "main", key, kinds); err != nil {
			return rt.Op{}, err
		}
		g.hl[pipe] = id
	}
	out := [3]string{"dummy1", "dummy2", "dummy2"}
	for i, d := range f.dests {
		bd, ok := batchDest[d]
		if !ok {
			return rt.Op{}, fmt.Errorf("%s: no batch buffer for %s", name, d)
		}
		out[i] = bd
	}
	switch len(f.dests) {
	case 1:
		out[1], out[2] = "dummy1", "dummy2"
	case 2:
		out[2] = "dummy1"
	}
	gid, err := g.dev.Group(name, pipe, []rt.Bind{
		{Buf: "bxq"}, {Buf: "bxs"}, {Buf: f.wn}, {Buf: f.mn}, {Buf: "bn", Size: 16},
		{Buf: out[0]}, {Buf: out[1]}, {Buf: out[2]},
	})
	if err != nil {
		return rt.Op{}, err
	}
	return rt.Op{Tag: rt.OpDispatch, Pipe: id, Group: gid, X: f.x, Y: f.y, Z: 1}, nil
}

// stepB runs ids (at most maxBatch) through every layer as one batch and
// leaves the last token's hidden state in x, where the head and the next
// decode step read it. With req the head runs in the same submit.
func (m *Model) stepB(d *dialog, ids []int, req *headReq) error {
	g := m.gpu
	b := g.batch
	n := len(ids)
	if n == 0 || n > maxBatch {
		return fmt.Errorf("batch: %d tokens", n)
	}
	if d.pos+n > g.ctx {
		return fmt.Errorf("context full (%d)", g.ctx)
	}
	le := binary.LittleEndian
	b.ids = b.ids[:0]
	b.ph = b.ph[:0]
	images := false
	for _, id := range ids {
		phi := float32(0)
		if sl, ok := m.slots[id]; ok {
			images = true
			phi = sl.phi
		} else if id < 0 || id >= m.Vocab {
			return fmt.Errorf("token %d is outside the embedding", id)
		}
		b.ids = le.AppendUint32(b.ids, uint32(id))
		b.ph = le.AppendUint32(b.ph, math.Float32bits(phi))
	}
	var step, cnt [16]byte
	le.PutUint32(step[0:], uint32(d.pos))
	le.PutUint32(cnt[0:], uint32(n))
	writes := []rt.Write{{Buf: b.bstep, Data: step[:]}, {Buf: b.bn, Data: cnt[:]}, {Buf: b.bids, Data: b.ids}, {Buf: b.bph, Data: b.ph}}
	if images {
		// Image tokens take their projected row; text rows of this buffer are not read.
		need := n * m.D * 4
		if cap(b.slot) < need {
			b.slot = make([]byte, need)
		}
		b.slot = b.slot[:need]
		for t, id := range ids {
			sl, ok := m.slots[id]
			if !ok {
				continue
			}
			row := b.slot[t*m.D*4:]
			for j, v := range sl.x {
				le.PutUint32(row[4*j:], math.Float32bits(v))
			}
		}
		writes = append(writes, rt.Write{Buf: b.bslot, Data: b.slot})
	}
	nn := uint32(n)
	ops := b.ops[:0]
	ops = append(ops, rt.Op{Tag: rt.OpDispatch, Pipe: b.embP, Group: b.embG, X: b.embX, Y: nn, Z: 1})
	for i := range b.layers {
		L := &b.layers[i]
		ops = append(ops,
			rt.Op{Tag: rt.OpDispatch, Pipe: b.normD, Group: L.an, X: nn, Y: 1, Z: 1},
			L.qkv,
			rt.Op{Tag: rt.OpDispatch, Pipe: b.ropeP, Group: L.rope, X: b.ropeX, Y: nn, Z: 1},
			rt.Op{Tag: rt.OpDispatch, Pipe: b.attnP, Group: L.attn, X: b.attnX, Y: nn, Z: 1},
			rt.Op{Tag: rt.OpDispatch, Pipe: b.normA, Group: L.sub, X: nn, Y: 1, Z: 1},
			L.o,
			rt.Op{Tag: rt.OpDispatch, Pipe: b.normD, Group: L.fn, X: nn, Y: 1, Z: 1},
			L.gu,
			rt.Op{Tag: rt.OpDispatch, Pipe: b.normF, Group: L.fs, X: nn, Y: 1, Z: 1},
			L.down,
		)
	}
	// The last token's residual stream becomes the one-token x.
	xid, err := g.dev.Buf("x", nil, 16)
	if err != nil {
		return err
	}
	ops = append(ops, rt.Op{Tag: rt.OpCopy, Src: b.bx, SrcOff: uint32((n - 1) * b.xBytes), Dst: xid, N: uint32(b.xBytes)})
	b.ops = ops
	g.ran(d, ids)
	if req != nil {
		return m.runHead(d, writes, ops, req)
	}
	_, err = g.dev.Run2(writes, ops, nil, false)
	return err
}
