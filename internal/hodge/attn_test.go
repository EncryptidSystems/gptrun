package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gptrun/internal/paths"
	"gptrun/internal/rt"
)

// TestAttnExact: attn.wgsl (and attnb.wgsl, token 0 of a batch) write the bits
// the kernels directory's attention.wgsl writes, at every cache length, and
// it reports the time of each.
func TestAttnExact(t *testing.T) {
	if _, err := os.Stat(rt.Exe()); err != nil {
		t.Skip(err)
	}
	orig, err := os.ReadFile(filepath.Join(paths.Default().KernelsDir(), "attention.wgsl"))
	if err != nil {
		t.Skip(err)
	}
	dev, err := rt.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	mine, _ := ownKernel("attn.wgsl")
	batch, _ := ownKernel("attnb.wgsl")
	grp, _ := ownKernel("attng.wgsl")
	grpb, _ := ownKernel("attngb.wgsl")
	for _, s := range []struct{ name, src string }{{"orig", string(orig)}, {"mine", mine}, {"batch", batch}, {"grp", grp}, {"grpb", grpb}} {
		if err := dev.Shader(s.name, s.src); err != nil {
			t.Fatal(s.name, err)
		}
	}
	const H, KV, HD, CTX = 20, 5, 128, 4096
	rng := rand.New(rand.NewSource(3))
	f32 := func(n int, sc float64) []byte {
		b := make([]byte, n*4)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(rng.NormFloat64()*sc)))
		}
		return b
	}
	bufs := []struct {
		name string
		data []byte
		n    int
	}{
		{"q", f32(H*HD, 0.3), 0}, {"kc", f32(CTX*KV*HD, 1), 0}, {"vc", f32(CTX*KV*HD, 1), 0},
		{"att", nil, H * CTX * 4}, {"o1", nil, H * HD * 4}, {"o2", nil, H * HD * 4}, {"o3", nil, H * HD * 4}, {"o4", nil, H * HD * 4}, {"gatt", nil, H * CTX * 4}, {"o5", nil, H * HD * 4}, {"gbatt", nil, H * CTX * 4},
		{"st", nil, 16}, {"nt", []byte{1, 0, 0, 0}, 16}, {"batt", nil, H * CTX * 4},
	}
	ids := map[string]uint32{}
	for _, b := range bufs {
		id, err := dev.Buf(b.name, b.data, b.n)
		if err != nil {
			t.Fatal(err)
		}
		ids[b.name] = id
	}
	over := fmt.Sprintf("H=%d KVH=%d HD=%d CTX=%d DENOM=0", H, KV, HD, CTX)
	k6 := []byte{rt.KindUni, rt.KindRO, rt.KindRO, rt.KindRO, rt.KindRW, rt.KindRW}
	k7 := append(append([]byte{}, k6...), rt.KindRO)
	po, _ := dev.Pipe("po", "orig", "main", over, k6)
	pm, _ := dev.Pipe("pm", "mine", "main", over, k6)
	pb, err := dev.Pipe("pb", "batch", "main", over, k7)
	if err != nil {
		t.Fatal(err)
	}
	pg, err := dev.Pipe("pg", "grp", "main", over, k6)
	if err != nil {
		t.Fatal(err)
	}
	g4, err := dev.Group("g4", "pg", []rt.Bind{{Buf: "st", Size: 16}, {Buf: "q"}, {Buf: "kc"}, {Buf: "vc"}, {Buf: "gatt"}, {Buf: "o4"}})
	if err != nil {
		t.Fatal(err)
	}
	pgb, err := dev.Pipe("pgb", "grpb", "main", over, k7)
	if err != nil {
		t.Fatal(err)
	}
	g5, err := dev.Group("g5", "pgb", []rt.Bind{{Buf: "st", Size: 16}, {Buf: "q"}, {Buf: "kc"}, {Buf: "vc"}, {Buf: "gbatt"}, {Buf: "o5"}, {Buf: "nt", Size: 16}})
	if err != nil {
		t.Fatal(err)
	}
	g1, _ := dev.Group("g1", "po", []rt.Bind{{Buf: "st", Size: 16}, {Buf: "q"}, {Buf: "kc"}, {Buf: "vc"}, {Buf: "att"}, {Buf: "o1"}})
	g2, _ := dev.Group("g2", "pm", []rt.Bind{{Buf: "st", Size: 16}, {Buf: "q"}, {Buf: "kc"}, {Buf: "vc"}, {Buf: "att"}, {Buf: "o2"}})
	g3, err := dev.Group("g3", "pb", []rt.Bind{{Buf: "st", Size: 16}, {Buf: "q"}, {Buf: "kc"}, {Buf: "vc"}, {Buf: "batt"}, {Buf: "o3"}, {Buf: "nt", Size: 16}})
	if err != nil {
		t.Fatal(err)
	}
	timeOps := func(op rt.Op) time.Duration {
		ops := make([]rt.Op, 30)
		for i := range ops {
			ops[i] = op
		}
		best := time.Duration(0)
		for range 3 {
			t0 := time.Now()
			if _, err := dev.Run(ops, nil); err != nil {
				t.Fatal(err)
			}
			if d := time.Since(t0); best == 0 || d < best {
				best = d
			}
		}
		return best
	}
	for _, n := range []int{1, 7, 8, 9, 17, 300, 1500, 4000} {
		rec := make([]byte, 16)
		binary.LittleEndian.PutUint32(rec, uint32(n-1))
		if err := dev.Write("st", 0, rec); err != nil {
			t.Fatal(err)
		}
		ops := []rt.Op{
			{Tag: rt.OpDispatch, Pipe: po, Group: g1, X: H, Y: 1, Z: 1},
			{Tag: rt.OpDispatch, Pipe: pm, Group: g2, X: H, Y: 1, Z: 1},
			{Tag: rt.OpDispatch, Pipe: pb, Group: g3, X: H, Y: 1, Z: 1},
			{Tag: rt.OpDispatch, Pipe: pg, Group: g4, X: KV, Y: 1, Z: 1},
			{Tag: rt.OpDispatch, Pipe: pgb, Group: g5, X: KV, Y: 1, Z: 1},
		}
		out, err := dev.Run(ops, []rt.Get{{Buf: ids["o1"], N: H * HD * 4}, {Buf: ids["o2"], N: H * HD * 4}, {Buf: ids["o3"], N: H * HD * 4}, {Buf: ids["o4"], N: H * HD * 4}, {Buf: ids["o5"], N: H * HD * 4}})
		if err != nil {
			t.Fatal(err)
		}
		sz := H * HD * 4
		for i := 0; i < sz; i += 4 {
			a := binary.LittleEndian.Uint32(out[i:])
			b := binary.LittleEndian.Uint32(out[sz+i:])
			c := binary.LittleEndian.Uint32(out[2*sz+i:])
			e := binary.LittleEndian.Uint32(out[3*sz+i:])
			f := binary.LittleEndian.Uint32(out[4*sz+i:])
			if a != b || a != c || a != e || a != f {
				t.Fatalf("n %d: output %d: orig %v, attn %v, attnb %v, attng %v, attngb %v", n, i/4,
					math.Float32frombits(a), math.Float32frombits(b), math.Float32frombits(c), math.Float32frombits(e), math.Float32frombits(f))
			}
		}
		if n >= 300 {
			to, tm, tg := timeOps(ops[0]), timeOps(ops[1]), timeOps(ops[3])
			t.Logf("n %4d: bit-identical; 30 layers: attention.wgsl %v, attn.wgsl %v, attng.wgsl %v", n, to.Round(time.Microsecond), tm.Round(time.Microsecond), tg.Round(time.Microsecond))
		}
	}
}
