package hodge

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

func TestProbeExpertDelta(t *testing.T) {
	if os.Getenv("HODGE_MOE") == "" {
		t.Skip()
	}
	const path = `C:\Users\grego\models\dev\hodge-5.1\hodge-5.1.bnz.ternary.safetensors`
	raw, err := readST(path, []string{
		"model.layers.0.mlp.experts.gate_up_proj",
		"model.layers.0.mlp.experts.gate_up_proj_scale",
		"model.layers.0.mlp.experts.gate_up_proj_bias",
		"model.layers.0.mlp.experts.down_proj",
		"model.layers.0.mlp.experts.down_proj_scale",
		"model.layers.0.mlp.experts.down_proj_bias",
		"model.layers.0.mlp.router.weight",
		"model.layers.0.mlp.router.bias",
	})
	if err != nil {
		t.Fatal(err)
	}
	const D, F, E = 2560, 6912, 128
	ff, _, err := loadMoE(raw, "model.layers.0.", D, F, 4, gptAlpha, gptLimit, nil, E)
	if err != nil {
		t.Fatal(err)
	}
	rms := func(v []float32) float64 {
		var s float64
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s / float64(len(v)))
	}
	t.Logf("router rms %.4g bias rms %.4g  gu scale %.4g down scale %.4g",
		rms(ff.router), rms(ff.rbias), ff.gu.s, ff.down.s)
	t.Logf("gu bias rms %.4g  down bias rms %.4g", rms(ff.guBias), rms(ff.dnBias))
	x := make([]float32, D)
	for i := range x {
		x[i] = 0.05
	}
	z := ff.delta(x)
	// The same forward with biases cleared, to see whether the grid or the bias speaks.
	keepG, keepD, keepR := append([]float32(nil), ff.guBias...), append([]float32(nil), ff.dnBias...), append([]float32(nil), ff.rbias...)
	keepW := append([]float32(nil), ff.router...)
	for i := range ff.guBias {
		ff.guBias[i] = 0
	}
	for i := range ff.dnBias {
		ff.dnBias[i] = 0
	}
	zb := ff.delta(x)
	ff.guBias, ff.dnBias = keepG, keepD
	for i := range ff.rbias {
		ff.rbias[i] = 0
	}
	for i := range ff.router {
		ff.router[i] = 0
	}
	zr := ff.delta(x)
	ff.rbias, ff.router = keepR, keepW
	var dot float64
	nz, nb := rms(z), rms(zb)
	for i := range z {
		dot += float64(z[i]) * float64(zb[i])
	}
	cos := dot / (nz * nb * float64(len(z)))
	t.Logf("x rms %.4g  delta %.4g  no-bias %.4g  cos %.3f  bias-only router %.4g", rms(x), nz, nb, cos, rms(zr))

	dense, err := readST(path, []string{
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.gate_proj.weight_scale",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.up_proj.weight_scale",
		"model.layers.0.mlp.down_proj.weight",
		"model.layers.0.mlp.down_proj.weight_scale",
	})
	if err != nil {
		t.Fatal(err)
	}
	sc := func(name string) float32 {
		v, err := bf16one(dense[name].b)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	g := ternY(dense["model.layers.0.mlp.gate_proj.weight"].b, sc("model.layers.0.mlp.gate_proj.weight_scale"), F, D, x)
	u := ternY(dense["model.layers.0.mlp.up_proj.weight"].b, sc("model.layers.0.mlp.up_proj.weight_scale"), F, D, x)
	h := make([]float32, F)
	for i := range h {
		if g[i] > 0 {
			h[i] = g[i] * g[i] * u[i]
		}
	}
	yd := ternY(dense["model.layers.0.mlp.down_proj.weight"].b, sc("model.layers.0.mlp.down_proj.weight_scale"), D, F, h)
	var c float64
	for i := range yd {
		c += float64(yd[i]) * float64(z[i])
	}
	t.Logf("dense relu2 rms %.4g  cos(expert, dense) %.3f", rms(yd), c/(rms(yd)*rms(z)*float64(D)))

	// RMSNorm output has rms 1. A constant 0.05 is not that vector.
	rng := uint32(1)
	xn := make([]float32, D)
	var ss float64
	for i := range xn {
		rng = rng*1664525 + 1013904223
		xn[i] = float32(int32(rng)) / (1 << 31)
		ss += float64(xn[i]) * float64(xn[i])
	}
	inv := float32(math.Sqrt(ss / float64(D)))
	for i := range xn {
		xn[i] /= inv
	}
	zn := ff.delta(xn)
	rng2 := uint32(99)
	x2 := make([]float32, D)
	ss = 0
	for i := range x2 {
		rng2 = rng2*1664525 + 1013904223
		x2[i] = float32(int32(rng2)) / (1 << 31)
		ss += float64(x2[i]) * float64(x2[i])
	}
	inv2 := float32(math.Sqrt(ss / float64(D)))
	for i := range x2 {
		x2[i] /= inv2
	}
	z2 := ff.delta(x2)
	var d12 float64
	for i := range zn {
		d12 += float64(zn[i]) * float64(z2[i])
	}
	t.Logf("unit-rms x  delta %.4g  second %.4g  cos %.3f", rms(zn), rms(z2), d12/(rms(zn)*rms(z2)*float64(D)))

	// Adjacent gate-up columns. Interleaved gate/up that survived the spline
	// disagree; a smoothed axis makes them the same vector.
	cols := ff.gu.cols
	col := func(c int) []float32 { return unpackCol(&ff.gu, 0, c) }
	var dotEU, nE, nU float64
	n := 0
	for c := 0; c+1 < cols; c += 2 {
		a, b := col(c), col(c+1)
		var d, aa, bb float64
		for i := range a {
			d += float64(a[i]) * float64(b[i])
			aa += float64(a[i]) * float64(a[i])
			bb += float64(b[i]) * float64(b[i])
		}
		dotEU += d
		nE += aa
		nU += bb
		n++
	}
	t.Logf("gate-up even/odd col cos %.3f over %d pairs", dotEU/math.Sqrt(nE*nU), n)
	have, err := layersWith(path, "mlp.gate_proj.weight")
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("layers with dense gate: %v", have)
	for _, li := range []int{0, 15, 29, 35} {
		p := fmt.Sprintf("model.layers.%d.", li)
		raw, err := readST(path, []string{
			p + "mlp.experts.gate_up_proj",
			p + "mlp.experts.gate_up_proj_scale",
			p + "mlp.experts.gate_up_proj_bias",
			p + "mlp.experts.down_proj_scale",
			p + "mlp.experts.down_proj_bias",
			p + "mlp.router.bias",
		})
		if err != nil {
			t.Fatal(err)
		}
		guS, _ := bf16one(raw[p+"mlp.experts.gate_up_proj_scale"].b)
		dnS, _ := bf16one(raw[p+"mlp.experts.down_proj_scale"].b)
		gb, _ := bf16s(raw[p+"mlp.experts.gate_up_proj_bias"].b)
		db, _ := bf16s(raw[p+"mlp.experts.down_proj_bias"].b)
		rb, _ := bf16s(raw[p+"mlp.router.bias"].b)
		t.Logf("L%d gu scale %.4g down scale %.4g  gu bias %.4g down bias %.4g router bias %.4g", li, guS, dnS, rms(gb), rms(db), rms(rb))
	}
	scN := []string{
		"model.layers.0.mlp.gate_proj.weight_scale",
		"model.layers.0.mlp.down_proj.weight_scale",
		"model.layers.30.mlp.gate_proj.weight_scale",
		"model.layers.30.mlp.down_proj.weight_scale",
		"model.layers.35.mlp.gate_proj.weight_scale",
		"model.layers.35.mlp.down_proj.weight_scale",
		"model.layers.30.self_attn.v_proj.weight_scale",
		"model.layers.30.self_attn.o_proj.weight_scale",
		"model.layers.0.mlp.ffn_sub_norm.weight",
		"model.layers.0.post_attention_layernorm.weight",
		"model.layers.0.input_layernorm.weight",
	}
	scT, err := readST(path, scN)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range scN {
		g := scT[name]
		if len(g.b) > 2 {
			v, err := bf16s(g.b)
			if err != nil {
				t.Fatal(err)
			}
			var s float64
			for _, x := range v {
				s += float64(x) * float64(x)
			}
			t.Logf("vec %s n %d rms %.6g", name, len(v), math.Sqrt(s/float64(len(v))))
			continue
		}
		v, err := bf16one(g.b)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("scale %s %.6g", name, v)
	}
	fmt.Fprintf(os.Stderr, "probe done\n")
}

func layersWith(path, suffix string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lb [8]byte
	if _, err := f.Read(lb[:]); err != nil {
		return nil, err
	}
	n := int64(binary.LittleEndian.Uint64(lb[:]))
	hdr := make([]byte, n)
	if _, err := f.Read(hdr); err != nil {
		return nil, err
	}
	var head map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &head); err != nil {
		return nil, err
	}
	var got []int
	for i := 0; i < 40; i++ {
		name := fmt.Sprintf("model.layers.%d.%s", i, suffix)
		if _, ok := head[name]; ok {
			got = append(got, i)
		}
	}
	return got, nil
}

func unpackCol(w *gridW, e, c int) []float32 {
	y := make([]float32, w.rows)
	np := w.rows / 4
	plane := np * w.cols
	add := func(buf []byte, scale float32, center int) {
		b := buf[e*plane : (e+1)*plane]
		for r := 0; r < np; r++ {
			by := b[r*w.cols+c]
			y[r] += float32(int(by&3)-center) * scale
			y[r+np] += float32(int((by>>2)&3)-center) * scale
			y[r+2*np] += float32(int((by>>4)&3)-center) * scale
			y[r+3*np] += float32(int((by>>6)&3)-center) * scale
		}
	}
	add(w.w, w.s, 1)
	for i := range w.limbs {
		lb := &w.limbs[i]
		add(lb.w, lb.s, lb.center)
	}
	return y
}

// ternY is y = W x for a packed ternary [rows, cols], limb 0 center 1.
func ternY(w []byte, scale float32, rows, cols int, x []float32) []float32 {
	y := make([]float32, rows)
	np := rows / 4
	for r := 0; r < np; r++ {
		row := w[r*cols : (r+1)*cols]
		for c := 0; c < cols; c++ {
			by := row[c]
			xc := x[c] * scale
			y[r] += float32(int(by&3)-1) * xc
			y[r+np] += float32(int((by>>2)&3)-1) * xc
			y[r+2*np] += float32(int((by>>4)&3)-1) * xc
			y[r+3*np] += float32(int((by>>6)&3)-1) * xc
		}
	}
	return y
}

func readST(path string, names []string) (map[string]got, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var lb [8]byte
	if _, err := f.Read(lb[:]); err != nil {
		return nil, err
	}
	n := int64(binary.LittleEndian.Uint64(lb[:]))
	hdr := make([]byte, n)
	if _, err := f.Read(hdr); err != nil {
		return nil, err
	}
	var head map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &head); err != nil {
		return nil, err
	}
	type tent struct {
		Dtype   string  `json:"dtype"`
		Shape   []int   `json:"shape"`
		Offsets []int64 `json:"data_offsets"`
	}
	out := map[string]got{}
	for _, name := range names {
		var e tent
		if err := json.Unmarshal(head[name], &e); err != nil {
			return nil, fmt.Errorf("%s: %v", name, err)
		}
		b := make([]byte, e.Offsets[1]-e.Offsets[0])
		if _, err := f.ReadAt(b, 8+n+e.Offsets[0]); err != nil {
			return nil, err
		}
		out[name] = got{b: b, shape: e.Shape}
	}
	return out, nil
}
