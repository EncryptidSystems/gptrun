package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"gptrun/internal/rt"
)

// TestKernelBench times one batched linear (hlxb.wgsl, gate+up shape, 32
// tokens) on synthetic data, for kernel work without loading a model.
// MODELRUN_KBENCH=1. MODELRUN_KBENCH_SRC is a WGSL file to time in its place,
// with the same bindings; one that reads xqf (f32 activations) gets f32 ones.
// MODELRUN_KBENCH_ROWS is the packed rows per workgroup of that file.
func TestKernelBench(t *testing.T) {
	if os.Getenv("MODELRUN_KBENCH") == "" {
		t.Skip("MODELRUN_KBENCH is not set")
	}
	dev, err := rt.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	src, err := ownKernel("hlxb.wgsl")
	if err != nil {
		t.Fatal(err)
	}
	if p := os.Getenv("MODELRUN_KBENCH_SRC"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		src = string(b)
	}
	if strings.Contains(src, "dot4(") {
		info, _, _ := dev.Info()
		dot := "fn dot4(a: u32, b: u32) -> i32 { var s = 0; for (var j = 0u; j < 4u; j++) { let sh = 24u - 8u * j; s += (bitcast<i32>(a << sh) >> 24u) * (bitcast<i32>(b << sh) >> 24u); } return s; }\n"
		if info&rt.InfoDot4 != 0 {
			dot = "requires packed_4x8_integer_dot_product;\nfn dot4(a: u32, b: u32) -> i32 { return dot4I8Packed(a, b); }\n"
		}
		src = dot + src
	}
	if err := dev.Shader("k", src); err != nil {
		t.Fatal(err)
	}
	const K, NP, NM, B = 2560, 1728, 2, 32
	rng := rand.New(rand.NewSource(1))
	w := make([]byte, NM*NP*K)
	rng.Read(w)
	var meta []byte
	le := binary.LittleEndian
	for m := 0; m < NM; m++ {
		for _, v := range []uint32{uint32(m * NP), NP, 1, 0, uint32(m * NP * K / 4), 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0} {
			meta = le.AppendUint32(meta, v)
		}
		for range maxPlanes {
			meta = le.AppendUint32(meta, math.Float32bits(0.01))
		}
	}
	xq := make([]byte, B*K*4)
	xi := make([]byte, B*K)
	for i := 0; i < B*K; i++ {
		v := int8(rng.Intn(256) - 128)
		xi[i] = byte(v)
		le.PutUint32(xq[i*4:], math.Float32bits(float32(v)))
	}
	xin := xi
	if strings.Contains(src, "xqf") {
		xin = xq
	}
	xs := make([]byte, B*4)
	for i := 0; i < B; i++ {
		le.PutUint32(xs[i*4:], math.Float32bits(0.02))
	}
	for _, b := range []struct {
		name string
		data []byte
		n    int
	}{
		{"xq", xin, len(xin)}, {"xs", xs, len(xs)}, {"w", w, len(w)}, {"meta", meta, len(meta)},
		{"o0", nil, B * NP * 16}, {"o1", nil, B * NP * 16}, {"o2", nil, 16},
	} {
		if _, err := dev.Buf(b.name, b.data, b.n); err != nil {
			t.Fatal(err)
		}
	}
	nb, _ := dev.Buf("nt", le.AppendUint32(nil, B), 16)
	_ = nb
	key := fmt.Sprintf("K=%d NM=%d MAXL=1 RESID=false", K, NM)
	kinds := []byte{rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW, rt.KindRW}
	p, err := dev.Pipe("p", "k", "main", key, kinds)
	if err != nil {
		t.Fatal(err)
	}
	gid, err := dev.Group("g", "p", []rt.Bind{{Buf: "xq"}, {Buf: "xs"}, {Buf: "w"}, {Buf: "meta"}, {Buf: "nt", Size: 16}, {Buf: "o0"}, {Buf: "o1"}, {Buf: "o2"}})
	if err != nil {
		t.Fatal(err)
	}
	rows := uint32(NM * NP)
	per := uint32(8)
	if v := os.Getenv("MODELRUN_KBENCH_ROWS"); v != "" {
		fmt.Sscan(v, &per)
	}
	x, y := grid((rows + per - 1) / per)
	ops := make([]rt.Op, 10)
	for i := range ops {
		ops[i] = rt.Op{Tag: rt.OpDispatch, Pipe: p, Group: gid, X: x, Y: y, Z: 1}
	}
	if _, err := dev.Run(ops[:1], nil); err != nil {
		t.Fatal(err)
	}
	var best time.Duration
	for range 5 {
		t0 := time.Now()
		if _, err := dev.Run(ops, nil); err != nil {
			t.Fatal(err)
		}
		if d := time.Since(t0); best == 0 || d < best {
			best = d
		}
	}
	per1 := best / 10
	wt := float64(NM*NP*K/4) * B
	t.Logf("%v per dispatch  %.1f G word-tokens/s  weights %.1f GB/s", per1, wt/per1.Seconds()/1e9, float64(len(w))/per1.Seconds()/1e9)
	out, err := dev.Run(nil, []rt.Get{{Buf: 5, N: 64}})
	if err == nil {
		t.Logf("o0[0..3] %v", f32s(out[:16]))
	}
}
