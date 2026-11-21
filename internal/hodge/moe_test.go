package hodge

import (
	"math"
	"os"
	"testing"

	"hodgemix/bnz4"
)

func TestGridSize(t *testing.T) {
	if gridSize(2560) != 160 || gridSize(6912) != 216 || gridSize(13824) != 216 {
		t.Fatalf("grids %d %d %d", gridSize(2560), gridSize(6912), gridSize(13824))
	}
}

func TestAxisMapInterp(t *testing.T) {
	a := newAxisMap(4, 2)
	got := a.up([]float32{10, 30})
	want := []float32{10, 15, 25, 30}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("up %v want %v", got, want)
		}
	}
	back := a.down(got)
	if math.Abs(float64(back[0]-27.5)) > 1e-3 || math.Abs(float64(back[1]-52.5)) > 1e-3 {
		t.Fatalf("down %v", back)
	}
}

func TestApplyCol(t *testing.T) {
	// rows 4, cols 2, codes:
	//   1 0
	//   2 1
	//   1 2
	//   0 0
	// packed NP=1: col0 codes 1,2,1,0 -> 0b00011001 = 25; col1 0,1,2,0 -> 0b00100100 = 36
	w := gridW{rows: 4, cols: 2, w: []byte{25, 36}, s: 2}
	y := w.applyCol(0, []float32{1, 0, 0, 0})
	if y[0] != 0 || y[1] != -2 {
		t.Fatalf("row0 %v", y)
	}
	y = w.applyCol(0, []float32{1, 1, 1, 1})
	if y[0] != 0 || y[1] != -2 {
		t.Fatalf("sum %v", y)
	}
}

func TestMoEDeltaBiasOnly(t *testing.T) {
	const D, F = 8, 8
	lifts := liftsFor(D, F)
	guRows, guCols := gridSize(D), gridSize(2*F)
	dnRows, dnCols := gridSize(F), gridSize(D)
	gu := gridW{rows: guRows, cols: guCols, w: bytesCenter(1, guRows, guCols), s: 1}
	dn := gridW{rows: dnRows, cols: dnCols, w: bytesCenter(1, dnRows, dnCols), s: 1}
	f := &moeFFN{
		experts: 1, topK: 1, D: D, F: F, two: 2 * F,
		alpha: gptAlpha, limit: gptLimit,
		gu: gu, down: dn,
		guBias: fill(2*F, 9),
		dnBias: fill(D, 3),
		router: make([]float32, D),
		rbias:  []float32{0},
		lift:   lifts,
	}
	z := f.delta(fill(D, 1))
	for i, v := range z {
		if math.Abs(float64(v-3)) > 1e-4 {
			t.Fatalf("z[%d]=%v", i, z)
		}
	}
}

func TestLoadMoENames(t *testing.T) {
	const D, F, E = 8, 8, 1
	guR, guC := gridSize(D), gridSize(2*F)
	dnR, dnC := gridSize(F), gridSize(D)
	p := "model.layers.0."
	raw := map[string]got{
		p + "mlp.experts.gate_up_proj":          {b: bytesCenter(E, guR, guC), shape: []int{E, guR / 4, guC}},
		p + "mlp.experts.gate_up_proj_scale":    {b: bf16bytes(1), shape: []int{1}},
		p + "mlp.experts.gate_up_proj_bias":     {b: bf16bytes(fill(E*2*F, 0)...), shape: []int{E, 2 * F}},
		p + "mlp.experts.down_proj":             {b: bytesCenter(E, dnR, dnC), shape: []int{E, dnR / 4, dnC}},
		p + "mlp.experts.down_proj_scale":       {b: bf16bytes(1), shape: []int{1}},
		p + "mlp.experts.down_proj_bias":        {b: bf16bytes(fill(E*D, 4)...), shape: []int{E, D}},
		p + "mlp.router.weight":                 {b: bf16bytes(make([]float32, E*D)...), shape: []int{E, D}},
		p + "mlp.router.bias":                   {b: bf16bytes(1), shape: []int{E}},
		p + "mlp.experts.down_proj_limb1":       {b: bytesCenter(E, dnR, dnC), shape: []int{E, dnR / 4, dnC}},
		p + "mlp.experts.down_proj_limb1_scale": {b: bf16bytes(0), shape: []int{1}},
		p + "mlp.experts.down_proj_limb1_radix": {b: []byte{0, 0, 0, 0}, shape: []int{1}},
	}
	ff, _, err := loadMoE(raw, p, D, F, 1, gptAlpha, gptLimit, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	z := ff.delta(fill(D, 0.2))
	for i, v := range z {
		if math.Abs(float64(v-4)) > 1e-3 {
			t.Fatalf("z[%d]=%v", i, z)
		}
	}
}

func TestLoadMoEWithoutShape(t *testing.T) {
	const D, F, E = 8, 8, 1
	guR, guC := gridSize(D), gridSize(2*F)
	dnR, dnC := gridSize(F), gridSize(D)
	p := "model.layers.0."
	raw := map[string]got{
		p + "mlp.experts.gate_up_proj":          {b: bytesCenter(E, guR, guC)},
		p + "mlp.experts.gate_up_proj_scale":    {b: bf16bytes(1)},
		p + "mlp.experts.gate_up_proj_bias":     {b: bf16bytes(fill(E*2*F, 0)...)},
		p + "mlp.experts.down_proj":             {b: bytesCenter(E, dnR, dnC)},
		p + "mlp.experts.down_proj_scale":       {b: bf16bytes(1)},
		p + "mlp.experts.down_proj_bias":        {b: bf16bytes(fill(E*D, 4)...)},
		p + "mlp.router.weight":                 {b: bf16bytes(make([]float32, E*D)...)},
		p + "mlp.router.bias":                   {b: bf16bytes(1)},
		p + "mlp.experts.down_proj_limb1":       {b: bytesCenter(E, dnR, dnC)},
		p + "mlp.experts.down_proj_limb1_scale": {b: bf16bytes(0)},
		p + "mlp.experts.down_proj_limb1_radix": {b: []byte{0, 0, 0, 0}},
	}
	ff, _, err := loadMoE(raw, p, D, F, 1, gptAlpha, gptLimit, nil, E)
	if err != nil {
		t.Fatal(err)
	}
	if ff.experts != E || ff.gu.rows != guR || ff.gu.cols != guC || ff.down.rows != dnR || ff.down.cols != dnC {
		t.Fatalf("grid %+v %+v", ff.gu, ff.down)
	}
	z := ff.delta(fill(D, 0.2))
	for i, v := range z {
		if math.Abs(float64(v-4)) > 1e-3 {
			t.Fatalf("z[%d]=%v", i, z)
		}
	}
}

func TestLoadMoEBNZLayer0(t *testing.T) {
	const path = `C:\Users\grego\models\dev\hodge-5\hodge-5-120B.bnz`
	if _, err := os.Stat(path); err != nil {
		t.Skip(err)
	}
	f, ix, base, err := openBNZ(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p := "model.layers.0."
	needName := map[string]bool{}
	for _, s := range []string{
		"mlp.experts.gate_up_proj",
		"mlp.experts.gate_up_proj_scale",
		"mlp.experts.gate_up_proj_bias",
		"mlp.experts.down_proj",
		"mlp.experts.down_proj_scale",
		"mlp.experts.down_proj_bias",
		"mlp.router.weight",
		"mlp.router.bias",
	} {
		needName[p+s] = true
	}
	raw := make(map[string]got, len(needName))
	for i := range ix.Segments {
		sg := &ix.Segments[i]
		if !needName[sg.Name] {
			continue
		}
		if len(sg.Shape) != 0 {
			t.Fatalf("%s shape %v", sg.Name, sg.Shape)
		}
		c := make([]byte, sg.Len)
		if _, err := f.ReadAt(c, base+sg.Off); err != nil {
			t.Fatal(sg.Name, err)
		}
		b, err := bnz4.DecodeSeg(sg, c)
		if err != nil {
			t.Fatal(err)
		}
		raw[sg.Name] = got{b: b, shape: append([]int(nil), sg.Shape...)}
	}
	if len(raw) != len(needName) {
		t.Fatalf("decoded %d of %d", len(raw), len(needName))
	}
	const D, F, E = 2560, 6912, 128
	ff, _, err := loadMoE(raw, p, D, F, 4, gptAlpha, gptLimit, nil, E)
	if err != nil {
		t.Fatal(err)
	}
	if ff.experts != E || ff.gu.rows != 160 || ff.gu.cols != 216 || ff.down.rows != 216 || ff.down.cols != 160 {
		t.Fatalf("experts %d gate-up %d×%d down %d×%d", ff.experts, ff.gu.rows, ff.gu.cols, ff.down.rows, ff.down.cols)
	}
	z := ff.delta(fill(D, 0.01))
	if len(z) != D {
		t.Fatalf("delta %d", len(z))
	}
}

func TestExpertChartRuns(t *testing.T) {
	p := "model.layers.0."
	chart := map[string]got{
		p + "mlp.experts.gate_up_proj":       {},
		p + "mlp.experts.gate_up_proj_scale": {},
		p + "mlp.gate_proj.weight":           {},
	}
	if runExperts(chart, p) {
		t.Fatal("scaled chart beside gate_proj left the dense block")
	}
	only := map[string]got{p + "mlp.experts.gate_up_proj": {}}
	if !runExperts(only, p) {
		t.Fatal("expert pack was skipped")
	}
	tail := map[string]got{
		p + "mlp.experts.gate_up_proj":       {},
		p + "mlp.experts.gate_up_proj_scale": {},
	}
	if !runExperts(tail, p) {
		t.Fatal("chart with no dense gate was skipped")
	}
	if runExperts(map[string]got{p + "mlp.gate_proj.weight": {}}, p) {
		t.Fatal("a layer with only the dense gate ran the chart")
	}
	body := make([]layer, 30)
	for i := range body {
		body[i].moe = &moeFFN{}
	}
	moe, mixed := moeFlags(body)
	if !moe || mixed {
		t.Fatalf("body chart moe %v mixed %v", moe, mixed)
	}
	// The tail cut drops expert layers that were counted before the slice.
	// The flags have to describe the layers that remain.
	moe, mixed = moeFlags(body[:30])
	if !moe || mixed {
		t.Fatalf("after cut moe %v mixed %v", moe, mixed)
	}
	body[29].moe = nil
	moe, mixed = moeFlags(body)
	if moe || !mixed {
		t.Fatalf("one dense layer moe %v mixed %v", moe, mixed)
	}
}

func TestLoadMoEUsesStoredGrid(t *testing.T) {
	const D, F, E = 8, 8, 1
	// Not the coarse index. The stored plane is the grid.
	guR, guC := 4, 6
	dnR, dnC := 4, 4
	p := "model.layers.0."
	raw := map[string]got{
		p + "mlp.experts.gate_up_proj":       {b: bytesCenter(E, guR, guC), shape: []int{E, guR / 4, guC}},
		p + "mlp.experts.gate_up_proj_scale": {b: bf16bytes(1), shape: []int{1}},
		p + "mlp.experts.gate_up_proj_bias":  {b: bf16bytes(fill(E*2*F, 0)...), shape: []int{E, 2 * F}},
		p + "mlp.experts.down_proj":          {b: bytesCenter(E, dnR, dnC), shape: []int{E, dnR / 4, dnC}},
		p + "mlp.experts.down_proj_scale":    {b: bf16bytes(1), shape: []int{1}},
		p + "mlp.experts.down_proj_bias":     {b: bf16bytes(fill(E*D, 4)...), shape: []int{E, D}},
		p + "mlp.router.weight":              {b: bf16bytes(make([]float32, E*D)...), shape: []int{E, D}},
		p + "mlp.router.bias":                {b: bf16bytes(1), shape: []int{E}},
		p + "mlp.gate_proj.weight":           {b: []byte{1}, shape: []int{1, 1}},
	}
	ff, _, err := loadMoE(raw, p, D, F, 1, gptAlpha, gptLimit, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if ff.gu.rows != guR || ff.gu.cols != guC || ff.down.rows != dnR || ff.down.cols != dnC {
		t.Fatalf("gate-up %d×%d down %d×%d", ff.gu.rows, ff.gu.cols, ff.down.rows, ff.down.cols)
	}
	if ff.lift.d.g != guR || ff.lift.two.g != guC || ff.lift.f.g != dnR || ff.lift.dout.g != dnC {
		t.Fatalf("lifts d %d two %d f %d dout %d", ff.lift.d.g, ff.lift.two.g, ff.lift.f.g, ff.lift.dout.g)
	}
	z := ff.delta(fill(D, 0.2))
	for i, v := range z {
		if math.Abs(float64(v-4)) > 1e-3 {
			t.Fatalf("z[%d]=%v", i, v)
		}
	}
}

func TestChartSubNorm(t *testing.T) {
	const D, F = 8, 8
	lifts := liftsFor(D, F)
	guRows, guCols := gridSize(D), gridSize(2*F)
	dnRows, dnCols := gridSize(F), gridSize(D)
	// Digit 2 is +1 at scale 1, so the down plane sums the gathered hidden.
	raw := bytesCenter(1, dnRows, dnCols)
	for i := range raw {
		raw[i] = 0xAA
	}
	ff := &moeFFN{
		experts: 1, topK: 1, D: D, F: F, two: 2 * F,
		alpha: gptAlpha, limit: gptLimit, eps: 1e-5,
		gu:     gridW{rows: guRows, cols: guCols, w: bytesCenter(1, guRows, guCols), s: 1},
		down:   gridW{rows: dnRows, cols: dnCols, w: raw, s: 1},
		guBias: fill(2*F, 4),
		dnBias: fill(D, 0),
		router: make([]float32, D),
		rbias:  []float32{0},
		lift:   lifts,
	}
	plain := vecRms(ff.delta(fill(D, 0.2)))
	ff.sub = fill(F, 1)
	if math.Abs(vecRms(ff.delta(fill(D, 0.2)))-plain) > 1e-4 {
		t.Fatal("unit sub-norm moved the chart")
	}
	ff.sub = fill(F, 0.5)
	got := vecRms(ff.delta(fill(D, 0.2)))
	if got >= plain*0.5 {
		t.Fatalf("sub-norm rms %.4g, plain %.4g", got, plain)
	}
}

func TestMoETopKTie(t *testing.T) {
	idx, w := moeTopK([]float32{1, 1, 0}, 2)
	if idx[0] != 0 || idx[1] != 1 {
		t.Fatalf("idx %v", idx)
	}
	if math.Abs(float64(w[0]-0.5)) > 1e-5 || math.Abs(float64(w[1]-0.5)) > 1e-5 {
		t.Fatalf("w %v", w)
	}
}

func bytesCenter(experts, rows, cols int) []byte {
	// Code 1 in every slot. With limb-0 center 1 that is a zero weight.
	b := make([]byte, experts*(rows/4)*cols)
	for i := range b {
		b[i] = 0x55
	}
	return b
}

func fill(n int, v float32) []float32 {
	o := make([]float32, n)
	for i := range o {
		o[i] = v
	}
	return o
}
