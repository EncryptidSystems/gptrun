package hodge

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"gptrun/internal/catalog"
	"gptrun/internal/chatfmt"
	"gptrun/internal/mark"
	"gptrun/internal/tokenizers"
)

// Turn is one chat message.
type Turn struct {
	Role    string
	Content string
}

// Opt is the sampler. ConfigSample fills any of Temp and TopP the caller left unset
// from the checkpoint's generation_config. Temp 0 is greedy. TempSet and TopSet
// mark a value the caller chose, including a temperature of 0.
type Opt struct {
	N            int
	Temp         float64
	TopP         float64
	ConfigSample bool
	TempSet      bool
	TopSet       bool
	// Halt ends the turn when it reports true for the text so far. The stop
	// token is not fed, so the same dialog can generate the next token.
	Halt func(string) bool
	// Image is an image file for the last user turn. Its vision tokens go at
	// the start of that turn's text (the checkpoint needs bnzmix.embed_vision).
	Image string
	// Memory is a file of passages ("---" between them) written into the
	// holonomy–eikonal field; what the field returns for the last user turn
	// goes in front of it (memfile.go).
	Memory string
}

type Model struct {
	name       string
	sys        string
	D, F, L    int
	H, KV, HD  int
	Vocab      int
	eps, theta float64
	act        string
	gemma      bool
	embedScale bool
	queryScale float64
	attnCap    float64
	finalCap   float64
	ropeScale  int
	ropeFactor float64
	ropeLow    float64
	ropeHigh   float64
	ropeOrig   float64
	temp, topP float64
	embed      []byte
	final      []float32
	layers     []layer
	freq       []float64
	gpu        *gpuRT
	tok        *tokenizers.Tokenizer
	tokPath    string
	tokCfg     string
	bos        []int
	stop       map[int]bool
	kind       chatfmt.Kind
	moe        bool            // every layer's feed-forward is the expert pack
	mixed      bool            // some layers are the dense block, some are the expert pack
	path       string          // the checkpoint file
	vis        *visionTab      // vision embedding layer, read when the first image comes
	slots      map[int]slotRow // installed image tokens, ids from Vocab up
}

// dialog is one conversation's place in the cache: ids[i] is the token at
// position i, and pos is len(ids). The device has one cache; gpuRT.kv says
// whose tokens it holds, and step replays a dialog's tokens another one wrote over.
type dialog struct {
	ids  []int
	pos  int
	lg   []float32
	hold bool // a halt paused the turn; the stop token was not fed
	full bool // the reply filled the window; transcriptHES slides it on
}

func systemText(name string) string {
	return "You are " + name + ", a helpful assistant that runs on this device. Talk with the person in plain words. When they want a recipe, give the ingredients and the method. When you need a lookup, a page, or a command, include one tool call in the reply: {\"type\":\"tool_call\",\"tool\":\"search\",\"args\":{\"q\":\"the question\"}}, {\"type\":\"tool_call\",\"tool\":\"gather\",\"args\":{\"url\":\"https://example.com\"}}, or {\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"ls\"}}. search looks something up. gather reads one public page. exec runs in the TunnelTug guest, never on this machine. After the results, answer in plain words. Do not call a tool for a greeting, an opinion, a story, or a fact you already know."
}

// ErrInterrupted is a turn stopped by Ctrl+C.
var ErrInterrupted = errors.New("interrupted")

var interrupted atomic.Bool

// Interrupt stops the token loop and the sandbox call.
func Interrupt() { interrupted.Store(true) }

// ClearInterrupt arms a new turn.
func ClearInterrupt() { interrupted.Store(false) }

// Interrupted reports a Ctrl+C since the last clear.
func Interrupted() bool { return interrupted.Load() }

// ropeOf reads rope_scaling the way rope.wgsl names it: 0 none, 1 linear, 2 llama3.
func ropeOf(raw json.RawMessage) (scale int, factor, low, high, orig float64) {
	factor, low, high, orig = 1, 1, 4, 8192
	if len(raw) == 0 || string(raw) == "null" {
		return 0, factor, low, high, orig
	}
	var o struct {
		Type   string  `json:"rope_type"`
		Factor float64 `json:"factor"`
		Low    float64 `json:"low_freq_factor"`
		High   float64 `json:"high_freq_factor"`
		Orig   float64 `json:"original_max_position_embeddings"`
	}
	if json.Unmarshal(raw, &o) != nil {
		return 0, factor, low, high, orig
	}
	switch o.Type {
	case "linear":
		scale = 1
	case "llama3":
		scale = 2
	}
	if o.Factor > 0 {
		factor = o.Factor
	}
	if o.Low > 0 {
		low = o.Low
	}
	if o.High > 0 {
		high = o.High
	}
	if o.Orig > 0 {
		orig = o.Orig
	}
	return scale, factor, low, high, orig
}

func open(meta catalog.Model) (*Model, error) {
	if meta.Config == "" || meta.Tokenizer == "" || meta.TokenizerConfig == "" {
		return nil, fmt.Errorf("%s: config, tokenizer, and tokenizer_config are set in the config file", meta.Name)
	}
	cfgb, err := os.ReadFile(meta.Config)
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Hidden     int             `json:"hidden_size"`
		Inter      int             `json:"intermediate_size"`
		Layers     int             `json:"num_hidden_layers"`
		Heads      int             `json:"num_attention_heads"`
		KV         int             `json:"num_key_value_heads"`
		HD         int             `json:"head_dim"`
		Vocab      int             `json:"vocab_size"`
		Eps        float64         `json:"rms_norm_eps"`
		Theta      float64         `json:"rope_theta"`
		Act        string          `json:"hidden_act"`
		Act2       string          `json:"hidden_activation"`
		Tie        *bool           `json:"tie_word_embeddings"`
		RMS        string          `json:"rms_norm"`
		QueryScale float64         `json:"query_pre_attn_scalar"`
		AttnCap    float64         `json:"attn_logit_softcapping"`
		FinalCap   float64         `json:"final_logit_softcapping"`
		BOS        *int            `json:"bos_token_id"`
		Rope       json.RawMessage `json:"rope_scaling"`
		Experts    int             `json:"num_local_experts"`
		Experts2   int             `json:"num_experts"`
		TopK       int             `json:"num_experts_per_tok"`
		TopK2      int             `json:"experts_per_token"`
		SwA        float64         `json:"swiglu_alpha"`
		SwL        float64         `json:"swiglu_limit"`
	}
	if err := json.Unmarshal(cfgb, &cfg); err != nil {
		return nil, err
	}
	if cfg.Tie != nil && !*cfg.Tie {
		return nil, fmt.Errorf("%s: Hodge ties the head to the embedding", meta.Name)
	}
	if cfg.KV == 0 {
		cfg.KV = cfg.Heads
	}
	if cfg.HD == 0 && cfg.Heads > 0 {
		cfg.HD = cfg.Hidden / cfg.Heads
	}
	if cfg.Hidden < 4 || cfg.Inter < 4 || cfg.Layers < 1 || cfg.Heads < 1 || cfg.Vocab < 1 || cfg.HD < 2 {
		return nil, fmt.Errorf("%s: config does not describe a hodge checkpoint", meta.Name)
	}
	if cfg.Heads%cfg.KV != 0 || cfg.Hidden%4 != 0 || cfg.Inter%4 != 0 || cfg.HD%2 != 0 {
		return nil, fmt.Errorf("%s: head or width is not a multiple the kernels accept", meta.Name)
	}
	act := cfg.Act
	if cfg.Act2 != "" {
		act = cfg.Act2
	}
	if act == "" {
		act = "relu2"
	}
	if gateFn(act) == nil {
		return nil, fmt.Errorf("%s: hidden activation %q is not in the runtime", meta.Name, act)
	}
	eps := cfg.Eps
	if eps == 0 {
		eps = 1e-5
	}
	theta := cfg.Theta
	if theta == 0 {
		theta = 500000
	}
	name := meta.Detail
	if name == "" {
		name = meta.Name
	}
	sys, err := resolveSystem(name)
	if err != nil {
		return nil, err
	}
	m := &Model{
		name: name, sys: sys, path: meta.Path,
		D: cfg.Hidden, F: cfg.Inter, L: cfg.Layers,
		H: cfg.Heads, KV: cfg.KV, HD: cfg.HD, Vocab: cfg.Vocab,
		eps: eps, theta: theta, act: act,
		gemma:      cfg.Act2 != "" || cfg.RMS == "gemma",
		queryScale: cfg.QueryScale, attnCap: cfg.AttnCap, finalCap: cfg.FinalCap,
		temp: 0.6, topP: 0.9, stop: map[int]bool{},
		tokPath: meta.Tokenizer, tokCfg: meta.TokenizerConfig,
	}
	m.ropeScale, m.ropeFactor, m.ropeLow, m.ropeHigh, m.ropeOrig = ropeOf(cfg.Rope)
	if cfg.BOS != nil && *cfg.BOS >= 0 {
		m.bos = []int{*cfg.BOS}
	}
	m.embedScale = m.gemma
	if meta.GenerationConfig != "" {
		b, err := os.ReadFile(meta.GenerationConfig)
		if err != nil {
			return nil, err
		}
		var g struct {
			EOS  json.RawMessage `json:"eos_token_id"`
			Temp float64         `json:"temperature"`
			TopP float64         `json:"top_p"`
		}
		if json.Unmarshal(b, &g) == nil {
			if g.Temp > 0 {
				m.temp = g.Temp
			}
			if g.TopP > 0 {
				m.topP = g.TopP
			}
			m.addEOS(g.EOS)
		}
	}
	logf("loading %s (%s)", meta.Name, name)
	raw, err := decodeAll(meta.Path)
	if err != nil {
		return nil, err
	}
	if layerCount(raw) != m.L {
		return nil, fmt.Errorf("%s: config says %d layers, the checkpoint has %d", meta.Name, m.L, layerCount(raw))
	}
	m.embed, err = assembleEmbed(raw, m.Vocab, m.D)
	if err != nil {
		return nil, err
	}
	if m.final, err = vec(raw, "model.norm.weight", m.D, false); err != nil {
		return nil, err
	}
	A := m.H * m.HD
	topK := cfg.TopK
	if topK == 0 {
		topK = cfg.TopK2
	}
	if topK == 0 {
		topK = gptTopK
	}
	alpha, limit := float32(gptAlpha), float32(gptLimit)
	if cfg.SwA > 0 {
		alpha = float32(cfg.SwA)
	}
	if cfg.SwL > 0 {
		limit = float32(cfg.SwL)
	}
	wantE := cfg.Experts
	if wantE == 0 {
		wantE = cfg.Experts2
	}
	var lifts *liftSet
	expertLayers := 0
	m.layers = make([]layer, m.L)
	for i := 0; i < m.L; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		L := &m.layers[i]
		if i == 0 {
			if g, ok := raw[p+"self_attn.sinks"]; ok {
				logf("attention sinks %v; this step uses %d heads", g.shape, m.H)
			}
		}
		if L.attnN, err = vec(raw, p+"input_layernorm.weight", m.D, false); err != nil {
			return nil, err
		}
		if L.ffnN, err = vec(raw, ffnNormName(raw, p), m.D, false); err != nil {
			return nil, err
		}
		if L.attnS, err = vec(raw, p+"self_attn.attn_sub_norm.weight", A, true); err != nil {
			return nil, err
		}
		if L.ffnS, err = vec(raw, p+"mlp.ffn_sub_norm.weight", m.F, true); err != nil {
			return nil, err
		}
		if L.qn, err = vec(raw, p+"self_attn.q_norm.weight", m.HD, true); err != nil {
			return nil, err
		}
		if L.kn, err = vec(raw, p+"self_attn.k_norm.weight", m.HD, true); err != nil {
			return nil, err
		}
		if L.q, err = buildLin(raw, p+"self_attn.q_proj", A, m.D); err != nil {
			return nil, err
		}
		if L.k, err = buildLin(raw, p+"self_attn.k_proj", m.KV*m.HD, m.D); err != nil {
			return nil, err
		}
		if L.v, err = buildLin(raw, p+"self_attn.v_proj", m.KV*m.HD, m.D); err != nil {
			return nil, err
		}
		if L.o, err = buildLin(raw, p+"self_attn.o_proj", m.D, A); err != nil {
			return nil, err
		}
		if runExperts(raw, p) {
			L.moe, lifts, err = loadMoE(raw, p, m.D, m.F, topK, alpha, limit, lifts, wantE)
			if err != nil {
				return nil, err
			}
			L.moe.sub = L.ffnS
			L.moe.eps = m.eps
			L.moe.gemma = m.gemma
			if expertLayers == 0 {
				logf("experts %d top %d  gate-up grid %d×%d  down grid %d×%d",
					L.moe.experts, L.moe.topK, L.moe.gu.rows, L.moe.gu.cols, L.moe.down.rows, L.moe.down.cols)
			}
			expertLayers++
		}
		// The dense gate is the relu2 block and stays loaded when it is
		// present. A layer with only the chart has no gate_proj.
		if _, ok := raw[p+"mlp.gate_proj.weight"]; ok || L.moe == nil {
			if L.gate, err = buildLin(raw, p+"mlp.gate_proj", m.F, m.D); err != nil {
				return nil, err
			}
			if L.up, err = buildLin(raw, p+"mlp.up_proj", m.F, m.D); err != nil {
				return nil, err
			}
			if L.down, err = buildLin(raw, p+"mlp.down_proj", m.D, m.F); err != nil {
				return nil, err
			}
		}
		switch {
		case L.moe != nil && (L.gate.w != nil || L.gate.f != nil):
			logf("layer %d  q %s  moe+dense", i, linKind(L.q))
		case L.moe != nil:
			logf("layer %d  q %s  moe", i, linKind(L.q))
		default:
			logf("layer %d  q %s  down %s", i, linKind(L.q), linKind(L.down))
		}
	}
	if cut := tailCut(qScales(m.layers)); cut < len(m.layers) {
		logf("layers %d-%d left out: query scale left the body (layer %d %.3g, layer 0 %.3g)",
			cut, m.L-1, cut, m.layers[cut].q.s, m.layers[0].q.s)
		m.layers = m.layers[:cut]
		m.L = cut
	}
	// Flags follow the layers that remain after the tail cut. A chart on every
	// layer is the mixture. A chart on only some of them is the mixed step.
	m.moe, m.mixed = moeFlags(m.layers)
	m.freq = make([]float64, m.HD/2)
	for d := range m.freq {
		m.freq[d] = math.Pow(m.theta, -2*float64(d)/float64(m.HD))
	}
	tok, err := tokenizers.Load(meta.Tokenizer)
	if err != nil {
		return nil, err
	}
	m.tok = tok
	if err := m.learnFormat(meta.TokenizerConfig); err != nil {
		return nil, err
	}
	if err := m.boot(); err != nil {
		return nil, err
	}
	logf("ready %s  %d layers  hidden %d", meta.Name, m.L, m.D)
	return m, nil
}

// stopIDs is the end tokens, in id order.
func (m *Model) stopIDs() []int {
	ids := make([]int, 0, len(m.stop))
	for id, on := range m.stop {
		if on {
			ids = append(ids, id)
		}
	}
	sort.Ints(ids)
	return ids
}

func (m *Model) addEOS(raw json.RawMessage) {
	if len(raw) == 0 {
		return
	}
	var n float64
	if json.Unmarshal(raw, &n) == nil {
		m.stop[int(n)] = true
		return
	}
	var arr []float64
	if json.Unmarshal(raw, &arr) == nil {
		for _, v := range arr {
			m.stop[int(v)] = true
		}
	}
}

func chatTemplates(path string) (string, []chatfmt.Listed) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", nil
	}
	var v any
	if json.Unmarshal(raw, &v) != nil {
		return "", nil
	}
	root, _ := v.(map[string]any)
	return chatfmt.ReadTemplates(root["chat_template"])
}

// learnFormat is the site's chat choice: an exact 2B4T template stays the role
// format even when the file also carries Llama 3 header pieces.
func (m *Model) learnFormat(tokenizerConfig string) error {
	primary, listed := chatTemplates(tokenizerConfig)
	has := func(piece string) bool { return m.tok.ID(piece) >= 0 }
	have := chatfmt.Available(has, primary, listed, "")
	m.kind = chatfmt.Plain
	if len(have) > 0 {
		m.kind = have[0]
	}
	picked, err := pickFormat(have)
	if err != nil {
		return err
	}
	if picked != "" {
		m.kind = picked
	}
	for _, piece := range chatfmt.Stops(m.kind) {
		if id := m.tok.ID(piece); id >= 0 {
			m.stop[id] = true
		}
	}
	logf("chat %s", chatfmt.Name(m.kind))
	return nil
}

func (m *Model) pieces(s string) []int {
	if s == "" {
		return nil
	}
	return m.tok.Encode(s, true)
}

func (m *Model) show(ids []int) string {
	if len(m.slots) == 0 {
		return m.tok.Decode(ids, true)
	}
	// Image tokens have no text; each run of them shows as one <image>.
	var b strings.Builder
	var run []int
	inImage := false
	for _, id := range ids {
		if m.isSlot(id) {
			if len(run) > 0 {
				b.WriteString(m.tok.Decode(run, true))
				run = run[:0]
			}
			if !inImage {
				b.WriteString("<image>")
			}
			inImage = true
			continue
		}
		inImage = false
		run = append(run, id)
	}
	if len(run) > 0 {
		b.WriteString(m.tok.Decode(run, true))
	}
	return b.String()
}

// withImage puts the image's tokens where the last user turn's text starts:
// the prompt rendered with that text empty shares everything before it.
func (m *Model) withImage(sys string, turns []Turn, ids []int, path string) ([]int, error) {
	slots, err := m.imageSlots(path)
	if err != nil {
		return nil, err
	}
	last := -1
	for i, t := range turns {
		if strings.EqualFold(t.Role, "user") {
			last = i
		}
	}
	if last < 0 {
		return nil, fmt.Errorf("an image needs a user turn")
	}
	empty := append([]Turn(nil), turns...)
	empty[last].Content = ""
	bare, err := m.render(sys, empty)
	if err != nil {
		return nil, err
	}
	// The kept turns may be a suffix of the history; align on the tail.
	at := commonPrefix(ids, bare)
	if at == 0 || at > len(ids) {
		return nil, fmt.Errorf("could not find where the user text starts")
	}
	out := make([]int, 0, len(ids)+len(slots))
	out = append(out, ids[:at]...)
	out = append(out, slots...)
	out = append(out, ids[at:]...)
	return out, nil
}

// turnIDs is one user turn the way hkruntime renders it. first carries BOS and the system text.
func turnIDs(tok *tokenizers.Tokenizer, kind chatfmt.Kind, bos []int, system, user string, first bool) []int {
	var ids []int
	if first {
		ids = append(ids, bos...)
	}
	return append(ids, tok.Encode(chatfmt.Render(kind, system, user, first), true)...)
}

func (m *Model) stopPiece() string {
	for _, piece := range chatfmt.Stops(m.kind) {
		if m.tok.ID(piece) >= 0 {
			return piece
		}
	}
	return ""
}

func quant(y []float32) ([]int8, float32) {
	mx := float32(1e-5)
	for _, v := range y {
		if a := float32(math.Abs(float64(v))); a > mx {
			mx = a
		}
	}
	s := 127 / mx
	q := make([]int8, len(y))
	for i, v := range y {
		q[i] = int8(math.Max(-128, math.Min(127, math.Round(float64(v*s)))))
	}
	return q, 1 / s
}

func rms(x, w []float32, eps float64, gemma bool) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := float32(1 / math.Sqrt(ss/float64(len(x))+eps))
	o := make([]float32, len(x))
	for i := range x {
		g := w[i]
		if gemma {
			g = 1 + g
		}
		o[i] = x[i] * inv * g
	}
	return o
}

func headNorm(x, w []float32, hd int, eps float64, gemma bool) {
	for h := 0; h+hd <= len(x); h += hd {
		copy(x[h:h+hd], rms(x[h:h+hd], w, eps, gemma))
	}
}

// qScales is each layer's limb-0 query scale.
func qScales(ls []layer) []float32 {
	s := make([]float32, len(ls))
	for i := range ls {
		s[i] = ls[i].q.s
	}
	return s
}

// tailCut is the first layer whose query scale left the body. A grown tail
// whose query is hundreds of times layer 0 (hodge-5.1 layers 30–35, scale
// 268–752) is not the token step. The body is kept.
func tailCut(scales []float32) int {
	if len(scales) < 2 || scales[0] == 0 {
		return len(scales)
	}
	lim := float32(math.Abs(float64(scales[0]))) * 8
	for i, s := range scales {
		if float32(math.Abs(float64(s))) > lim {
			return i
		}
	}
	return len(scales)
}

// runExperts reports that this layer's FFN is the quaternary chart.
// The dense gate is the relu2 block. A chart stored beside that gate is not
// added into the residual. A layer with only the chart uses the chart.
func runExperts(t map[string]got, prefix string) bool {
	if _, dense := t[prefix+"mlp.gate_proj.weight"]; dense {
		return false
	}
	_, ok := t[prefix+"mlp.experts.gate_up_proj"]
	return ok
}

// moeFlags is the step taken by the layers that are actually resident.
func moeFlags(layers []layer) (moe, mixed bool) {
	n := 0
	for i := range layers {
		if layers[i].moe != nil {
			n++
		}
	}
	return n == len(layers) && n > 0, n > 0 && n < len(layers)
}

// ffnNormName is the FFN pre-norm. Llama and Qwen store it as post_attention_layernorm.
// VaultGemma stores that same pre-norm as pre_feedforward_layernorm.
func ffnNormName(t map[string]got, prefix string) string {
	post := prefix + "post_attention_layernorm.weight"
	if _, ok := t[post]; ok {
		return post
	}
	pre := prefix + "pre_feedforward_layernorm.weight"
	if _, ok := t[pre]; ok {
		return pre
	}
	return post
}

func linKind(l lin) string {
	if l.f != nil {
		if len(l.bias) > 0 {
			return "float+bias"
		}
		return "float"
	}
	return fmt.Sprintf("limbs %d", len(l.limbs))
}

// actID is norm.wgsl's ACT override. 0 relu^2, 1 silu, 2 gelu tanh.
func actID(act string) (int, error) {
	switch act {
	case "relu2":
		return 0, nil
	case "silu", "swish":
		return 1, nil
	case "gelu_pytorch_tanh", "gelu_new":
		return 2, nil
	default:
		return 0, fmt.Errorf("hidden activation %q is not in norm.wgsl", act)
	}
}

func gateFn(act string) func(float32) float32 {
	switch act {
	case "relu2":
		return func(g float32) float32 {
			if g < 0 {
				g = 0
			}
			return g * g
		}
	case "silu", "swish":
		return func(g float32) float32 { return g / (1 + float32(math.Exp(float64(-g)))) }
	case "gelu_pytorch_tanh", "gelu_new":
		return func(g float32) float32 {
			x := float64(g)
			return float32(0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
		}
	default:
		return nil
	}
}

func (m *Model) row(id int) []float32 {
	off := id * m.D * 2
	o := make([]float32, m.D)
	for j := 0; j < m.D; j++ {
		o[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(m.embed[off+2*j:])) << 16)
	}
	if m.embedScale {
		sc := float32(math.Sqrt(float64(m.D)))
		for j := range o {
			o[j] *= sc
		}
	}
	return o
}

func (m *Model) rope(x []float32, pos int) {
	hd := m.HD
	half := hd / 2
	for h := 0; h+hd <= len(x); h += hd {
		for d := 0; d < half; d++ {
			ang := float64(pos) * m.freq[d]
			c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
			a, b := x[h+d], x[h+d+half]
			x[h+d] = a*c - b*s
			x[h+d+half] = a*s + b*c
		}
	}
}

func (m *Model) linear(l lin, q []int8, xs float32) ([]float32, error) {
	return nil, fmt.Errorf("linear %d is on the device", l.n)
}

func (m *Model) attend(q []float32, kc, vc [][]float32) []float32 {
	H, KV, HD := m.H, m.KV, m.HD
	ao := make([]float32, H*HD)
	T := len(kc)
	den := math.Sqrt(float64(HD))
	if m.queryScale > 0 {
		den = math.Sqrt(m.queryScale)
	}
	for h := 0; h < H; h++ {
		g := h / (H / KV)
		sc := make([]float64, T)
		mx := -1e30
		for t := 0; t < T; t++ {
			var a float64
			for d := 0; d < HD; d++ {
				a += float64(q[h*HD+d] * kc[t][g*HD+d])
			}
			a /= den
			if m.attnCap > 0 {
				a = math.Tanh(a/m.attnCap) * m.attnCap
			}
			sc[t] = a
			if a > mx {
				mx = a
			}
		}
		var z float64
		for t := range sc {
			sc[t] = math.Exp(sc[t] - mx)
			z += sc[t]
		}
		for t := range sc {
			w := float32(sc[t] / z)
			for d := 0; d < HD; d++ {
				ao[h*HD+d] += w * vc[t][g*HD+d]
			}
		}
	}
	return ao
}

func (m *Model) logits(x []float32) []float32 {
	y := rms(x, m.final, m.eps, m.gemma)
	out := make([]float32, m.Vocab)
	emb := m.embed
	D := m.D
	workers := 8
	var wg sync.WaitGroup
	for p := 0; p < workers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for r := p; r < len(out); r += workers {
				var a float32
				off := r * D * 2
				for j := 0; j < D; j++ {
					a += y[j] * math.Float32frombits(uint32(binary.LittleEndian.Uint16(emb[off+2*j:]))<<16)
				}
				out[r] = a
			}
		}(p)
	}
	wg.Wait()
	if m.finalCap > 0 {
		cap := m.finalCap
		for i, v := range out {
			out[i] = float32(math.Tanh(float64(v)/cap) * cap)
		}
	}
	return out
}

// feed runs one token. With req it also leaves the next logits in d.lg.
func (m *Model) feed(d *dialog, id int, req *headReq) error {
	if id < 0 || id >= m.Vocab {
		return fmt.Errorf("token %d is outside the embedding", id)
	}
	return m.step(d, []int{id}, req)
}

func (m *Model) prefill(d *dialog, ids []int) error {
	if len(ids) == 0 {
		return fmt.Errorf("empty prompt")
	}
	tail := ids
	if len(tail) > 24 {
		tail = tail[len(tail)-24:]
	}
	logf("prompt %d  tail %q", len(ids), m.show(tail))
	// Tokens whose cache rows the device already holds, for this same
	// sequence, are not run again. The last token always runs: the head reads
	// its hidden state.
	if g := m.gpu; g != nil {
		full := append(append([]int(nil), d.ids[:d.pos]...), ids...)
		// Rows this dialog would write over go to the shelf, and its own rows
		// come back from it.
		if err := m.fit(full); err != nil {
			return err
		}
		c := min(commonPrefix(g.kv, full), len(full)-1)
		if c > d.pos {
			logf("reusing %d cached tokens", c-d.pos)
			d.ids = append(d.ids[:d.pos], full[d.pos:c]...)
			d.pos = c
			ids = full[c:]
		}
	}
	logf("prefill %d tokens", len(ids))
	t0 := time.Now()
	// The logits come with the first draw (generate), which knows the sampler.
	if err := m.step(d, ids, nil); err != nil {
		return err
	}
	// The submits above do not wait. Waiting here costs one round trip and keeps
	// the prefill's time out of the decode rate.
	if err := m.sync(); err != nil {
		return err
	}
	logf("prefill %d/%d  %s", len(ids), len(ids), time.Since(t0).Round(time.Millisecond))
	return nil
}

func commonPrefix(a, b []int) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func emitText(pending *[]byte, piece string, emit func(string)) {
	if emit == nil {
		return
	}
	*pending = append(*pending, piece...)
	i := 0
	for i < len(*pending) {
		r, size := utf8.DecodeRune((*pending)[i:])
		if r == utf8.RuneError && size == 1 {
			break
		}
		i += size
	}
	if i > 0 {
		emit(string((*pending)[:i]))
		*pending = (*pending)[i:]
	}
}

// Tests set these: a fixed sampling seed, and the full head on every token.
var (
	testSeed  int64
	testExact bool
)

func (m *Model) generate(d *dialog, opt Opt, emit func(string)) (string, error) {
	// N <= 0 runs until an end token or the context window. A positive N is a cap.
	n := opt.N
	if n <= 0 {
		n = 1 << 30
	}
	temp, top := opt.Temp, opt.TopP
	if opt.ConfigSample {
		if !opt.TempSet {
			temp = m.temp
		}
		if !opt.TopSet {
			top = m.topP
		}
	}
	var rng *rand.Rand
	if temp > 0 {
		seed := time.Now().UnixNano()
		if testSeed != 0 {
			seed = testSeed
		}
		rng = rand.New(rand.NewSource(seed))
	}
	var pending []byte
	var all []byte
	var recent []int
	ended := false
	d.hold = false
	d.full = false
	emitted := 0
	t0 := time.Now()
	defer func() { noteSpeed(emitted, time.Since(t0).Seconds()) }()
	keepID := func(id int) bool {
		return m.stop[id] || (m.tok != nil && m.tok.IsSpecial(id))
	}
	ends := m.stopIDs()
	req := &headReq{temp: temp, keep: keepID, exact: testExact}
	if rng == nil {
		req.temp = 0
	}
	for j := 0; j < n; j++ {
		if interrupted.Load() {
			return string(all), ErrInterrupted
		}
		if m.gpu != nil && d.pos >= m.gpu.ctx {
			d.full = true
			break
		}
		if d.lg == nil {
			req.recent = recent
			if err := m.head(d, req); err != nil {
				return "", err
			}
		}
		id := draw(d.lg, recent, temp, top, keepID, ends, rng)
		if m.stop[id] {
			ended = true
			if err := m.feed(d, id, nil); err != nil {
				return "", err
			}
			break
		}
		emitted++
		piece := string(m.tok.Bytes(id))
		all = append(all, piece...)
		recent = remember(recent, id, repeatLast)
		emitText(&pending, piece, emit)
		// The next logits come in the same submit as this token, unless this
		// is the last token the cap allows or the turn is pausing here.
		halt := opt.Halt != nil && opt.Halt(string(all))
		var next *headReq
		if !halt && j+1 < n {
			req.recent = recent
			next = req
		}
		if err := m.feed(d, id, next); err != nil {
			return "", err
		}
		if halt {
			d.hold = true
			break
		}
	}
	if len(pending) > 0 && emit != nil {
		emit(string(pending))
	}
	if !ended && !d.hold && !d.full {
		if end := m.stopPiece(); end != "" {
			ids := m.pieces(end)
			if err := m.step(d, ids, nil); err != nil {
				return "", err
			}
		}
	}
	secs := time.Since(t0).Seconds()
	if secs > 0 {
		logf("generated in %s", time.Since(t0).Round(time.Millisecond))
	}
	return string(all), nil
}

func (m *Model) refuse(texts ...string) error {
	c, err := mark.Cached(m.tokPath, m.tokCfg)
	if err != nil || c == nil {
		return err
	}
	for _, text := range texts {
		if err := c.Refuse(text); err != nil {
			return err
		}
	}
	return nil
}

func (m *Model) begin(turns []Turn, opt Opt) (*dialog, error) {
	turns = dropGuardrails(turns)
	var texts []string
	for _, t := range turns {
		if strings.EqualFold(t.Role, "system") {
			continue
		}
		texts = append(texts, t.Content)
	}
	if err := m.refuse(texts...); err != nil {
		return nil, err
	}
	sys := systemFor(m)
	for _, t := range turns {
		if strings.EqualFold(t.Role, "system") && strings.TrimSpace(t.Content) != "" {
			sys = t.Content
		}
	}
	// A history longer than the window loses its oldest turns, as Ollama
	// does with num_ctx. Those turns are written into a field first, and one
	// basin is read back when the leftover tokens can hold it.
	ids, kept, dropped, err := m.fitTurns(sys, turns, m.ctxLen(), room(opt))
	if errors.Is(err, ErrTooLong) && opt.Image == "" {
		// The last user turn alone is longer than the window: its head goes
		// into the holonomy–eikonal field and comes back as recalled basins.
		long, oerr := m.overflowTurn(sys, turns, m.ctxLen()-max(room(opt), min(opt.N, m.ctxLen()/2)))
		if oerr != nil {
			return nil, err
		}
		turns = long
		ids, kept, dropped, err = m.fitTurns(sys, turns, m.ctxLen(), room(opt))
	}
	if err != nil {
		return nil, err
	}
	if dropped > 0 {
		logf("history trimmed  %d old turns dropped", dropped)
		ids, kept = m.withDroppedHES(sys, turns, kept, m.ctxLen()-room(opt))
	}
	if opt.Memory != "" {
		// The recalled basins may use the window up to the whole reply the run
		// asked for (-n), not just the default reply room.
		window := m.ctxLen() - max(room(opt), opt.N)
		if opt.Image != "" {
			window -= visionSlots
		}
		if ids, kept, err = m.withMemoryHES(sys, kept, ids, opt.Memory, window); err != nil {
			return nil, err
		}
	}
	if hesFile != "" {
		window := m.ctxLen() - max(room(opt), opt.N)
		if opt.Image != "" {
			window -= visionSlots
		}
		ids, kept = m.withChatHES(sys, kept, ids, window)
	}
	lastRun.sys, lastRun.kept, lastRun.h = sys, append([]Turn(nil), kept...), nil
	if opt.Image != "" {
		if dropped > 0 {
			return nil, fmt.Errorf("an image needs the whole conversation in the window")
		}
		if ids, err = m.withImage(sys, kept, ids, opt.Image); err != nil {
			return nil, err
		}
		if len(ids)+room(opt) > m.ctxLen() {
			return nil, fmt.Errorf("%w (%d tokens with the image; the window holds %d with room for the reply)", ErrTooLong, len(ids), m.ctxLen()-room(opt))
		}
	}
	d := &dialog{}
	if err := m.prefill(d, ids); err != nil {
		return nil, err
	}
	return d, nil
}

func (m *Model) transcript(turns []Turn, opt Opt, emit func(string)) (string, error) {
	d, err := m.begin(turns, opt)
	if err != nil {
		return "", err
	}
	text, err := m.transcriptHES(d, opt, emit)
	if err == nil && hesFile != "" {
		if li := lastUser(turns); li >= 0 {
			m.noteHES(turns[li].Content, text)
		}
	}
	return text, err
}

// transcriptContinue generates with first, and when that pass pauses and
// goOn accepts the text, generates with second on the same dialog. The
// pause does not feed the stop token, so the second pass is the next token.
func (m *Model) transcriptContinue(turns []Turn, first, second Opt, goOn func(string) bool, bridge string, emit1, emit2 func(string)) (string, string, bool, error) {
	d, err := m.begin(turns, first)
	if err != nil {
		return "", "", false, err
	}
	thought, err := m.generate(d, first, emit1)
	if err != nil || !d.hold || goOn == nil || !goOn(thought) {
		return thought, "", false, err
	}
	// The cue is a few tokens on the thought cache. It is not a second prefill
	// of the thought, and it is not shown. The answer sampler then runs.
	if bridge != "" {
		if err := m.step(d, m.pieces(bridge), nil); err != nil {
			return thought, "", false, err
		}
	}
	d.lg = nil
	answer, err := m.generate(d, second, emit2)
	return thought, answer, true, err
}

func (m *Model) raw(prompt string, opt Opt, emit func(string)) (string, error) {
	if err := m.refuse(prompt); err != nil {
		return "", err
	}
	d := &dialog{}
	ids := append(append([]int{}, m.bos...), m.tok.Encode(prompt, true)...)
	if err := m.prefill(d, ids); err != nil {
		return "", err
	}
	return m.generate(d, opt, emit)
}

var (
	liveMu   sync.Mutex
	live     *Model
	liveName string
	chat     *dialog
	chatName string
)

func get(name string) (*Model, error) {
	if live != nil {
		if liveName == name {
			return live, nil
		}
		return nil, fmt.Errorf("%s is loaded; quit and run the other model", liveName)
	}
	meta, err := catalog.Load(name)
	if err != nil {
		return nil, err
	}
	if meta.Family != catalog.FamilyHodge {
		return nil, fmt.Errorf("no hodge model %q", name)
	}
	m, err := open(meta)
	if err != nil {
		return nil, err
	}
	live, liveName = m, name
	dropChat()
	return m, nil
}

// Prepare loads the checkpoint and leaves the weights on the device.
func Prepare(name string) error {
	liveMu.Lock()
	defer liveMu.Unlock()
	_, err := get(name)
	return err
}

// Warm reads system prompts into the cache before the first turn, so it does
// not wait on them: each of side (a stateless pass such as the route pass),
// then the chat's own. Their rows stay on the shelf and in the cache, and a
// turn takes them from there instead of reading them again.
func Warm(name string, side ...string) error {
	liveMu.Lock()
	defer liveMu.Unlock()
	return do(name, func(m *Model) error {
		return m.warm(append(side, systemFor(m)))
	}, nil)
}

// warm prefills, for each system in order, the tokens every first turn under
// it begins with. The last one stays in the cache; the one before it is on the
// shelf.
func (m *Model) warm(systems []string) error {
	if m.gpu == nil {
		return nil
	}
	t0 := time.Now()
	n := 0
	for _, sys := range systems {
		// What a first turn under sys starts with, whatever the person says.
		a := turnIDs(m.tok, m.kind, m.bos, sys, "a", true)
		b := turnIDs(m.tok, m.kind, m.bos, sys, "Why is the sky blue?", true)
		p := commonPrefix(a, b)
		if p < 2 {
			continue
		}
		if err := m.prefill(&dialog{}, a[:p]); err != nil {
			return err
		}
		n += p
	}
	logf("warm  %d tokens  %s", n, time.Since(t0).Round(time.Millisecond))
	return nil
}

var skillBody string

// UseSkill arms one skill for the next chat. An empty body turns it off.
// The chat cache is dropped so the next turn sees the skill. The weights stay.
func UseSkill(body string) {
	liveMu.Lock()
	skillBody = strings.TrimSpace(body)
	dropChat()
	liveMu.Unlock()
}

func systemFor(m *Model) string {
	if skillBody == "" {
		return m.sys
	}
	return m.sys + "\n\n" + skillBody
}

// Reset drops the chat cache. The checkpoint stays loaded.
func Reset(name string) {
	liveMu.Lock()
	defer liveMu.Unlock()
	if name == "" || name == chatName {
		dropChat()
	}
}

// TranscriptContinue is one stateless chat that may keep generating on the
// same cache after Halt pauses the first pass.
func TranscriptContinue(name string, turns []Turn, first, second Opt, goOn func(string) bool, bridge string, emit1, emit2 func(string)) (thought, answer string, continued bool, err error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	emit1, quiet1 := watchEmit(emit1)
	emit2, quiet2 := watchEmit(emit2)
	err = do(name, func(m *Model) error {
		var e error
		thought, answer, continued, e = m.transcriptContinue(turns, first, second, goOn, bridge, emit1, emit2)
		return e
	}, func() bool { return quiet1() && quiet2() })
	return thought, answer, continued, err
}

// Transcript is one stateless chat. turns keeps the client's history.
func Transcript(name string, turns []Turn, opt Opt, emit func(string)) (string, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	emit, quiet := watchEmit(emit)
	var text string
	err := do(name, func(m *Model) (err error) {
		text, err = m.transcript(turns, opt, emit)
		return err
	}, quiet)
	return text, err
}

// Raw is a completion of prompt, without the chat template.
func Raw(name, prompt string, opt Opt, emit func(string)) (string, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	emit, quiet := watchEmit(emit)
	var text string
	err := do(name, func(m *Model) (err error) {
		text, err = m.raw(prompt, opt, emit)
		return err
	}, quiet)
	return text, err
}

// Absorb writes one finished exchange into the chat cache. A looked-up answer
// is the page's own sentences, so it is stored here and the model is not asked
// to rewrite it.
func Absorb(name, user, assistant string) error {
	liveMu.Lock()
	defer liveMu.Unlock()
	return do(name, func(m *Model) error {
		if err := m.refuse(user, assistant); err != nil {
			return err
		}
		if chat == nil || chatName != name {
			dropChat()
			chat, chatName = &dialog{}, name
		}
		reply := m.pieces(assistant)
		if end := m.stopPiece(); end != "" {
			reply = append(reply, m.pieces(end)...)
		}
		ids, err := m.nextTurnAt(user, len(reply), false)
		if err != nil {
			return err
		}
		if err := m.prefill(chat, append(ids, reply...)); err != nil {
			return err
		}
		chatLog = append(chatLog, Turn{Role: "user", Content: user}, Turn{Role: "assistant", Content: assistant})
		m.noteHES(user, assistant)
		return nil
	}, nil)
}

// Continue is the TUI conversation. The first call for a model starts a new chat.
func Continue(name, user string, opt Opt, emit func(string)) (string, error) {
	liveMu.Lock()
	defer liveMu.Unlock()
	emit, quiet := watchEmit(emit)
	var text string
	err := do(name, func(m *Model) error {
		if chat == nil || chatName != name {
			dropChat()
			chat, chatName = &dialog{}, name
		}
		if err := m.refuse(user); err != nil {
			return err
		}
		ids, err := m.nextTurn(user, room(opt))
		if err != nil {
			return err
		}
		mark := chat.pos
		if err := m.prefill(chat, ids); err != nil {
			chat.rewind(mark)
			return err
		}
		t, err := m.generate(chat, opt, emit)
		text = t
		if guardrail(text) {
			chat.rewind(mark)
		} else if err == nil || errors.Is(err, ErrInterrupted) {
			// An interrupted reply stays in the cache, so it stays in the log.
			said := user
			if promptedUser != "" {
				said = promptedUser
			}
			chatLog = append(chatLog, Turn{Role: "user", Content: said}, Turn{Role: "assistant", Content: text})
			m.noteHES(user, text)
		}
		return err
	}, quiet)
	return text, err
}

// Loaded is the checkpoint held in this process, if any.
func Loaded() (string, bool) {
	liveMu.Lock()
	defer liveMu.Unlock()
	if live == nil {
		return "", false
	}
	return liveName, true
}
