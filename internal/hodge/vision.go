package hodge

// Images. A checkpoint with bnzmix.embed_vision carries the L-A vision tower
// (hodgemix/tower, the package tower.wasm is built from, so the features are
// the ones the projector was fitted to) and a projector from its readout into
// the residual stream. An image becomes tower.VisSlots-wide reserved token ids
// one past the text embedding: at those positions the projected vector is the
// layer-0 state, and the token's phase is added to rotary pairs 56-63 of q and
// k. That is hodgemix-go/eval (embed, rope, install) and EdgeAI's media path.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"math/cmplx"
	"os"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/webp"

	"hodgemix/tower"
)

// visionTab is the vision embedding layer: projector [D][df], the feature mean
// it was centred on, the mean row and the embedding rms it matches.
type visionTab struct {
	D, df   int
	W       []float32
	meanF   []float32
	meanRow []float32
	rms     float64
}

// visionSlots is how many tokens an image takes.
const visionSlots = tower.VisSlots

// slotRow is one image token: its layer-0 state and its rotary phase.
type slotRow struct {
	x   []float32
	phi float32
}

// stRaw is one tensor read straight from a safetensors file.
type stRaw struct {
	dtype string
	shape []int
	b     []byte
}

// readNamed reads the named tensors from a safetensors file. Missing names are
// left out of the map.
func readNamed(path string, want func(string) bool) (map[string]stRaw, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var nb [8]byte
	if _, err := io.ReadFull(f, nb[:]); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	n := binary.LittleEndian.Uint64(nb[:])
	if n == 0 || n > 1<<30 {
		return nil, fmt.Errorf("%s is not a safetensors checkpoint", path)
	}
	hj := make([]byte, n)
	if _, err := io.ReadFull(f, hj); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var hdr map[string]json.RawMessage
	if err := json.Unmarshal(hj, &hdr); err != nil {
		return nil, fmt.Errorf("%s is not a safetensors checkpoint", path)
	}
	base := int64(8 + n)
	out := map[string]stRaw{}
	for name, raw := range hdr {
		if name == "__metadata__" || !want(name) {
			continue
		}
		var info struct {
			Dtype string  `json:"dtype"`
			Shape []int   `json:"shape"`
			Off   []int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &info); err != nil || len(info.Off) != 2 || info.Off[1] < info.Off[0] {
			return nil, fmt.Errorf("%s: %s: bad header", path, name)
		}
		b := make([]byte, info.Off[1]-info.Off[0])
		if _, err := f.ReadAt(b, base+info.Off[0]); err != nil {
			return nil, fmt.Errorf("%s: %s: %w", path, name, err)
		}
		out[name] = stRaw{dtype: info.Dtype, shape: info.Shape, b: b}
	}
	return out, nil
}

func stF32(t map[string]stRaw, name string, n int) ([]float32, error) {
	r, ok := t[name]
	if !ok {
		return nil, fmt.Errorf("checkpoint has no %s", name)
	}
	var o []float32
	switch r.dtype {
	case "F32":
		o = make([]float32, len(r.b)/4)
		for i := range o {
			o[i] = math.Float32frombits(binary.LittleEndian.Uint32(r.b[4*i:]))
		}
	case "BF16":
		o = make([]float32, len(r.b)/2)
		for i := range o {
			o[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(r.b[2*i:])) << 16)
		}
	default:
		return nil, fmt.Errorf("%s is %s", name, r.dtype)
	}
	if n > 0 && len(o) != n {
		return nil, fmt.Errorf("%s has %d values, want %d", name, len(o), n)
	}
	return o, nil
}

// loadVision reads the vision embedding layer and checks that the file's tower
// constants are the L-A tower this build runs (as hodgemix-go/eval checkTower).
func loadVision(path string, D int) (*visionTab, error) {
	pre := tower.VisEmbed
	t, err := readNamed(path, func(n string) bool {
		return len(n) > len(pre) && n[:len(pre)] == pre || len(n) > len(tower.VisTower) && n[:len(tower.VisTower)] == tower.VisTower
	})
	if err != nil {
		return nil, err
	}
	w, ok := t[pre+".weight"]
	if !ok {
		return nil, fmt.Errorf("this checkpoint has no vision embedding layer (%s.weight)", pre)
	}
	if !tower.Ready() {
		return nil, fmt.Errorf("the L-A vision tower is not ready")
	}
	sig, err := stF32(t, tower.VisTower+".sigma", 2)
	if err != nil {
		return nil, err
	}
	zs, err := stF32(t, tower.VisTower+".zstar", 2)
	if err != nil {
		return nil, err
	}
	if cmplx.Abs(complex(float64(sig[0]), float64(sig[1]))-tower.Sigma()) > 2e-5 || cmplx.Abs(complex(float64(zs[0]), float64(zs[1]))-tower.ZStar) > 2e-5 {
		return nil, fmt.Errorf("the file's vision tower constants are not the L-A tower this build runs")
	}
	if d, ok := t[tower.VisTower+".depth"]; !ok || len(d.b) < 4 || int32(binary.LittleEndian.Uint32(d.b)) != tower.Depth {
		return nil, fmt.Errorf("the file's vision tower depth is not %d", tower.Depth)
	}
	var W []float32
	var rows, df int
	switch w.dtype {
	case "U8":
		if len(w.shape) != 2 {
			return nil, fmt.Errorf("%s.weight shape %v", pre, w.shape)
		}
		rows, df = w.shape[0]*4, w.shape[1]
		sc, err := stF32(t, pre+".weight_scale", 1)
		if err != nil {
			return nil, err
		}
		var limbs []tower.Limb
		for i := 1; ; i++ {
			name := fmt.Sprintf("%s.weight_limb%d", pre, i)
			l, ok := t[name]
			if !ok {
				break
			}
			ls, err := stF32(t, name+"_scale", 1)
			if err != nil {
				return nil, err
			}
			rx, ok := t[name+"_radix"]
			if !ok || len(rx.b) != 4 || len(l.shape) != 2 || l.shape[0] != rows/4 || l.shape[1] != df {
				return nil, fmt.Errorf("%s: incomplete limb", name)
			}
			limbs = append(limbs, tower.Limb{Radix: int(binary.LittleEndian.Uint32(rx.b)), Digits: l.b, Scale: ls[0]})
		}
		W = tower.Dense(w.b, rows, df, sc[0], limbs)
	default:
		if len(w.shape) != 2 {
			return nil, fmt.Errorf("%s.weight shape %v", pre, w.shape)
		}
		rows, df = w.shape[0], w.shape[1]
		if W, err = stF32(t, pre+".weight", rows*df); err != nil {
			return nil, err
		}
	}
	if rows != D {
		return nil, fmt.Errorf("vision projector width %d, model hidden %d", rows, D)
	}
	if df != tower.VisFeat {
		return nil, fmt.Errorf("vision projector takes %d features, the tower gives %d", df, tower.VisFeat)
	}
	mf, err := stF32(t, pre+".mean_feature", df)
	if err != nil {
		return nil, err
	}
	my, err := stF32(t, pre+".mean_row", D)
	if err != nil {
		return nil, err
	}
	r, err := stF32(t, pre+".rms", 1)
	if err != nil {
		return nil, err
	}
	return &visionTab{D: D, df: df, W: W, meanF: mf, meanRow: my, rms: float64(r[0])}, nil
}

// imageRGB decodes an image file into row-major RGB in [0, 1], as the tower takes it.
func imageRGB(path string) ([]float32, int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("%s: %w", path, err)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 1 || h < 1 {
		return nil, 0, 0, fmt.Errorf("%s: empty image", path)
	}
	rgb := make([]float32, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := 3 * (y*w + x)
			rgb[i] = float32(r) / 65535
			rgb[i+1] = float32(g) / 65535
			rgb[i+2] = float32(bl) / 65535
		}
	}
	return rgb, w, h, nil
}

// imageSlots encodes an image file into reserved token ids one past the text
// embedding, replacing any image installed before. The cached rows are dropped
// so a different image under the same ids is not reused.
func (m *Model) imageSlots(path string) ([]int, error) {
	if m.vis == nil {
		v, err := loadVision(m.path, m.D)
		if err != nil {
			return nil, err
		}
		m.vis = v
		logf("vision  projector [%d %d]  tower %d features", v.D, v.df, tower.VisFeat)
	}
	rgb, w, h, err := imageRGB(path)
	if err != nil {
		return nil, err
	}
	feats, phi, err := tower.Vision(rgb, w, h)
	if err != nil {
		return nil, err
	}
	m.slots = map[int]slotRow{}
	ids := make([]int, len(feats))
	for i, f := range feats {
		x, err := tower.Project(m.vis.W, m.vis.D, m.vis.df, m.vis.meanF, m.vis.meanRow, m.vis.rms, f)
		if err != nil {
			return nil, err
		}
		p := float32(0)
		if i < len(phi) {
			p = float32(phi[i])
		}
		id := m.Vocab + i
		m.slots[id] = slotRow{x: x, phi: p}
		ids[i] = id
	}
	if m.gpu != nil {
		m.gpu.kv = nil
	}
	logf("image %s  %dx%d  %d tokens", path, w, h, len(ids))
	return ids, nil
}

// isSlot reports whether id is an installed image token.
func (m *Model) isSlot(id int) bool {
	_, ok := m.slots[id]
	return ok
}
