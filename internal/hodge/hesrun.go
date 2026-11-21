package hodge

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"

	"gptrun/internal/hes"
	"gptrun/internal/rt"
)

// The runtime's memory is the whole holonomy–eikonal state (internal/hes):
// keys from Hodge's own embedding rows map each write onto the latent torus
// (φ), each write marches with its token class's speed field, salience and
// the recency head start compete in one min, both holonomy channels develop,
// and reads follow the characteristic back to the basin. The GPU window holds
// the recent tokens; everything else lives in the field, so a prompt longer
// than the window, a history longer than the window and a reply longer than
// the window all go through it instead of failing with "context full".
//
// HES state files (-hes FILE) keep the chat field across runs.

// hesFile is the state file the chat field loads from and saves to.
var hesFile string

// chatHES is this chat's field.
var chatHES *hes.HES

// UseHESFile keeps the chat's holonomy–eikonal field in path across runs. An
// empty path keeps it in memory only.
func UseHESFile(path string) {
	liveMu.Lock()
	hesFile = path
	chatHES = nil
	liveMu.Unlock()
}

// hesSeed fixes the torus, φ and the speed fields, so a saved state rebuilds
// the same metric.
const hesSeed = 0x48455321

// chatWindow is the retention target W of the chat field: the design rule
// picks λ so about this many recent writes stay retrievable.
const chatWindow = 512

// chunkTokens is the size of one write when a long text goes into the field.
const chunkTokens = 192

// hesDims is the torus of every field: 64 × 64 = 4096 vertices, the size of
// the window the field stands in for. HES_DIMS=48x48x2 or similar overrides.
func hesDims() []int {
	if s := os.Getenv("HES_DIMS"); s != "" {
		var out []int
		for _, p := range strings.Split(strings.ToLower(s), "x") {
			var n int
			if _, err := fmt.Sscan(p, &n); err == nil && n > 0 {
				out = append(out, n)
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return []int{64, 64}
}

// --- keys --------------------------------------------------------------

var (
	embMu   sync.Mutex
	embMean []float32
	embFor  *Model
)

// embCache holds embedding rows read back from the device.
var embCache = map[int][]float32{}

const embCacheMax = 1 << 16

func hesRow(b []byte, d int) []float32 {
	o := make([]float32, d)
	for j := 0; j < d; j++ {
		o[j] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(b[2*j:])) << 16)
	}
	return o
}

// embRows is the embedding row of each id. The host copy is used while the
// model has one; once the weights are resident (gpu.go drops the host copy)
// the rows are read back from the device's embedding chunks.
func (m *Model) embRows(ids []int) (map[int][]float32, error) {
	out := make(map[int][]float32, len(ids))
	rowBytes := m.D * 2
	if m.embed != nil {
		for _, id := range ids {
			if id >= 0 && id < m.Vocab {
				out[id] = hesRow(m.embed[id*rowBytes:], m.D)
			}
		}
		return out, nil
	}
	g := m.gpu
	if g == nil {
		return nil, fmt.Errorf("no embedding on the host or the device")
	}
	embMu.Lock()
	defer embMu.Unlock()
	var want []int
	seen := map[int]bool{}
	for _, id := range ids {
		if id < 0 || id >= m.Vocab || seen[id] {
			continue
		}
		seen[id] = true
		if r, ok := embCache[id]; ok {
			out[id] = r
			continue
		}
		want = append(want, id)
	}
	for len(want) > 0 {
		n := min(len(want), 1024)
		batch := want[:n]
		want = want[n:]
		gets := make([]rt.Get, 0, n)
		for _, id := range batch {
			c, local, err := g.chunk(id)
			if err != nil {
				return nil, err
			}
			gets = append(gets, rt.Get{Buf: c.buf, Off: local * rowBytes, N: rowBytes})
		}
		b, err := g.dev.Run2(nil, nil, gets, true)
		if err != nil {
			return nil, err
		}
		if len(embCache) > embCacheMax {
			embCache = map[int][]float32{}
		}
		for i, id := range batch {
			r := hesRow(b[i*rowBytes:], m.D)
			out[id] = r
			embCache[id] = r
		}
	}
	return out, nil
}

// vocabMean is the mean embedding row, subtracted from every key: the common
// direction of all rows carries no content and would put every key in the
// middle of the torus. It is estimated from 8192 rows drawn evenly.
func (m *Model) vocabMean() []float32 {
	embMu.Lock()
	if embFor == m && embMean != nil {
		defer embMu.Unlock()
		return embMean
	}
	embMu.Unlock()
	n := min(8192, m.Vocab)
	ids := make([]int, n)
	for i := range ids {
		ids[i] = i * m.Vocab / n
	}
	rows, err := m.embRows(ids)
	mean := make([]float32, m.D)
	if err == nil && len(rows) > 0 {
		acc := make([]float64, m.D)
		for _, r := range rows {
			for j, v := range r {
				acc[j] += float64(v)
			}
		}
		for j := range mean {
			mean[j] = float32(acc[j] / float64(len(rows)))
		}
	} else if err != nil {
		logf("hes vocabulary mean  %v", err)
	}
	embMu.Lock()
	embMean, embFor = mean, m
	embMu.Unlock()
	return mean
}

// keyOf is the write's key: the mean of its tokens' embedding rows (specials
// left out, as EmbedDB does), less the vocabulary mean, at unit length.
func (m *Model) keyOf(ids []int) []float32 {
	if len(ids) == 0 || (m.embed == nil && m.gpu == nil) {
		return nil
	}
	var use []int
	for _, id := range ids {
		if id >= 0 && id < m.Vocab && (m.tok == nil || !m.tok.IsSpecial(id)) {
			use = append(use, id)
		}
	}
	if len(use) == 0 {
		return nil
	}
	rows, err := m.embRows(use)
	if err != nil {
		logf("hes key  %v", err)
		return nil
	}
	D := m.D
	acc := make([]float64, D)
	n := 0
	for _, id := range use {
		r, ok := rows[id]
		if !ok {
			continue
		}
		for j, v := range r {
			acc[j] += float64(v)
		}
		n++
	}
	if n == 0 {
		return nil
	}
	mean := m.vocabMean()
	key := make([]float32, D)
	var ss float64
	for j := range acc {
		v := acc[j]/float64(n) - float64(mean[j])
		key[j] = float32(v)
		ss += v * v
	}
	if ss == 0 {
		return nil
	}
	inv := float32(1 / math.Sqrt(ss))
	for j := range key {
		key[j] *= inv
	}
	return key
}

// sampleKeys calibrates φ's spread for a field whose writes are not known in
// advance: keys of random 32-token draws from the vocabulary.
func (m *Model) sampleKeys(n int) [][]float32 {
	rng := rand.New(rand.NewSource(hesSeed))
	out := make([][]float32, 0, n)
	for tries := 0; len(out) < n && tries < 4*n; tries++ {
		ids := make([]int, 32)
		for i := range ids {
			ids[i] = rng.Intn(m.Vocab)
		}
		if k := m.keyOf(ids); k != nil {
			out = append(out, k)
		}
	}
	return out
}

// --- fields ------------------------------------------------------------

// ensureHES is the chat field: loaded from the state file (and its key
// statistics from STATE.keys) when there is one, otherwise new.
func (m *Model) ensureHES() *hes.HES {
	if chatHES != nil {
		return chatHES
	}
	if hesFile != "" {
		h, err := hes.LoadHES(hesFile)
		switch {
		case err == nil && h.K != nil && h.K.M != m.keyDim():
			logf("hes %s holds keys of %d values, this runtime makes %d (starting a new field)", hesFile, h.K.M, m.keyDim())
		case err == nil:
			chatHES = h
			if st, err := loadStats(hesFile + ".keys"); err == nil {
				fieldStats[h] = st
			}
			logf("hes loaded %s  %d writes  %d payloads  λ %d", hesFile, h.Writes(), len(h.F.Payload), h.Lambda())
			return h
		case !os.IsNotExist(err):
			logf("hes %s: %v (starting a new field)", hesFile, err)
		}
	}
	h, err := newChatField()
	if err != nil {
		logf("hes off  %v", err)
		return nil
	}
	chatHES = h
	m.logLaws("chat", h)
	return h
}

func (m *Model) saveHES() {
	if hesFile == "" || chatHES == nil {
		return
	}
	if err := chatHES.Save(hesFile); err != nil {
		logf("hes save %s: %v", hesFile, err)
		return
	}
	if err := saveStats(hesFile+".keys", statsOf(chatHES)); err != nil {
		logf("hes save %s.keys: %v", hesFile, err)
	}
}

func (m *Model) logLaws(what string, h *hes.HES) {
	r := h.Laws()
	logf("hes %s  N %d (D %d)  λ %d  Λ %.3g  %s  half-life %d (universal %.0f)  capacity %.0f  sandwich %d–%d  writes %d  live %d/%d",
		what, r.N, r.D, r.Lambda, r.Scaled, r.Regime, r.HalfLifeExact, r.HalfLifeUniversal, r.Capacity,
		r.SandwichLo, r.SandwichHi, r.Writes, r.Live, r.Payloads)
}

func (m *Model) logH1(what string, h *hes.HES) {
	cls := h.Persistence(h.G.Base, 8)
	obs := 0
	for _, c := range cls {
		if c.Obstruction {
			obs++
		}
	}
	logf("hes %s  H1 %d classes kept  %d obstructions", what, len(cls), obs)
}

// writeText puts text into h as chunk writes and returns how many.
func (m *Model) writeText(h *hes.HES, text string) int {
	n := 0
	for _, c := range m.chunks(text) {
		info, err := m.writeKeyed(h, c, m.pieces(c))
		if err != nil {
			logf("hes write  %v", err)
			continue
		}
		n++
		logf("hes write  vertex %d  class %d  birth %d  salience %d  took %d  holonomy %s",
			info.Vertex, info.Class, info.Birth, info.Salience, info.Took, info.Hol)
	}
	return n
}

// chunks cuts text into writes of about chunkTokens: paragraphs, then
// sentences, then token runs.
func (m *Model) chunks(text string) []string {
	text = strings.TrimSpace(strings.ReplaceAll(text, "\r\n", "\n"))
	if text == "" {
		return nil
	}
	var units []string
	for _, para := range strings.Split(text, "\n\n") {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}
		if len(m.pieces(para)) <= chunkTokens {
			units = append(units, para)
			continue
		}
		for _, s := range sentences(para) {
			ids := m.pieces(s)
			for len(ids) > chunkTokens {
				units = append(units, m.tok.Decode(ids[:chunkTokens], true))
				ids = ids[chunkTokens:]
			}
			if len(ids) > 0 {
				units = append(units, m.tok.Decode(ids, true))
			}
		}
	}
	var out []string
	cur, curN := "", 0
	for _, u := range units {
		n := len(m.pieces(u))
		if cur != "" && curN+n > chunkTokens {
			out = append(out, cur)
			cur, curN = "", 0
		}
		if cur == "" {
			cur = u
		} else {
			cur += "\n\n" + u
		}
		curN += n
	}
	if cur != "" {
		out = append(out, cur)
	}
	return out
}

func sentences(s string) []string {
	cut := strings.NewReplacer(". ", ".\n", "? ", "?\n", "! ", "!\n", "\n", "\n")
	var out []string
	for _, x := range strings.Split(cut.Replace(s), "\n") {
		if x = strings.TrimSpace(x); x != "" {
			out = append(out, x)
		}
	}
	return out
}

// recallFor reads h for the user turn: the whole turn and each of its
// sentences are queries, and every payload scores its earliest continued
// arrival over them. The order is the field's.
func (m *Model) recallFor(h *hes.HES, user string, skip func(string) bool) []hes.Recall {
	best := map[string]hes.Recall{}
	for _, q := range queryViews(user) {
		ids := m.pieces(q)
		for _, r := range h.ReadAll(m.keyFor(h, q, ids), ids, skip, 0) {
			b, ok := best[r.Payload]
			if !ok || better(r, b) {
				best[r.Payload] = r
			}
		}
	}
	out := make([]hes.Recall, 0, len(best))
	for _, r := range best {
		out = append(out, r)
	}
	// Landmark probes first, by rank; then the owner of φ(q), the read itself
	// (−∇T from q); then the continued arrivals of the other payloads.
	sort.Slice(out, func(i, j int) bool {
		if better(out[i], out[j]) != better(out[j], out[i]) {
			return better(out[i], out[j])
		}
		if out[i].Owner != out[j].Owner {
			return out[i].Owner
		}
		if out[i].Arrival != out[j].Arrival {
			return out[i].Arrival < out[j].Arrival
		}
		return out[i].Basin < out[j].Basin
	})
	return out
}

// better orders two reads of the same payload or of two payloads: a landmark
// probe (lower rank first) before the owner of φ(q) before the continued
// arrival (sooner first).
func better(a, b hes.Recall) bool {
	pa, pb := a.Probe >= 0, b.Probe >= 0
	if pa != pb {
		return pa
	}
	if pa && a.Probe != b.Probe {
		return a.Probe < b.Probe
	}
	if a.Owner != b.Owner {
		return a.Owner
	}
	return a.Arrival < b.Arrival
}

// packRecall puts as many whole recalled payloads as the window takes in
// front of the last user turn (in field order), or the longest prefix of the
// first when not even that fits. order, when set, reorders what was taken
// (a document's own order).
func (m *Model) packRecall(sys string, kept []Turn, li int, got []hes.Recall, window int, order func([]string) []string) ([]int, []Turn, int) {
	user := kept[li].Content
	render := func(content string) ([]int, error) {
		trial := append([]Turn(nil), kept...)
		trial[li].Content = content
		return m.render(sys, trial)
	}
	var taken []string
	joinOf := func(ps []string) string {
		if order != nil {
			ps = order(append([]string(nil), ps...))
		}
		return strings.Join(ps, "\n\n")
	}
	for _, r := range got {
		trial, err := render(hes.Prompt(user, joinOf(append(taken, r.Payload))))
		if err != nil || len(trial) > window {
			continue // a shorter basin later in the order may still fit
		}
		taken = append(taken, r.Payload)
		logf("hes recall  basin %d  entry %d  arrival %d  owner %v  holonomy %s  %q", r.Basin, r.Vertex, r.Arrival, r.Owner, r.Hol, snippet(r.Payload))
	}
	joined := joinOf(taken)
	if joined == "" && len(got) > 0 {
		base, err := render(user)
		if err != nil {
			return nil, kept, 0
		}
		nTok := func(s string) int {
			t, err := render(s)
			if err != nil {
				return window + 1
			}
			return len(t)
		}
		joined = hes.Fit(user, got[0].Payload, window-len(base), nTok)
		if joined != "" {
			taken = []string{joined}
		}
	}
	if joined == "" {
		return nil, kept, 0
	}
	out := append([]Turn(nil), kept...)
	out[li].Content = hes.Prompt(user, joined)
	ids, err := m.render(sys, out)
	if err != nil || len(ids) > window {
		return nil, kept, 0
	}
	return ids, out, len(taken)
}

// lastParagraph is the text after the last blank line.
func lastParagraph(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.LastIndex(s, "\n\n"); i >= 0 {
		if p := strings.TrimSpace(s[i+2:]); p != "" {
			return p
		}
	}
	return s
}

func snippet(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if r := []rune(s); len(r) > 70 {
		return string(r[:70]) + "…"
	}
	return s
}

func lastUser(turns []Turn) int {
	for i := len(turns) - 1; i >= 0; i-- {
		if strings.EqualFold(turns[i].Role, "user") {
			return i
		}
	}
	return -1
}

// --- the run paths -------------------------------------------------------

// withMemoryHES is -memory: the file's passages are written into a library
// field and read back for the last user turn.
func (m *Model) withMemoryHES(sys string, kept []Turn, ids []int, path string, window int) ([]int, []Turn, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	passages := splitPassages(string(b))
	if len(passages) == 0 {
		return ids, kept, nil
	}
	li := lastUser(kept)
	if li < 0 {
		return nil, nil, fmt.Errorf("memory needs a user turn")
	}
	h, err := m.libraryFor(passages)
	if err != nil {
		return nil, nil, err
	}
	m.logLaws("memory", h)
	m.logH1("memory", h)
	user := kept[li].Content
	got := m.recallFor(h, user, func(p string) bool { return p == "" || p == user })
	nids, out, n := m.packRecall(sys, kept, li, got, window, nil)
	logf("hes memory  %d passages written  %d read  %d in the window", len(passages), len(got), n)
	if n == 0 {
		return ids, kept, nil
	}
	return nids, out, nil
}

// withChatHES reads the persistent chat field (-hes) for the last user turn
// of a stateless run and puts what fits in front of it.
func (m *Model) withChatHES(sys string, kept []Turn, ids []int, window int) ([]int, []Turn) {
	h := m.ensureHES()
	li := lastUser(kept)
	if h == nil || h.Writes() == 0 || li < 0 {
		return ids, kept
	}
	user := kept[li].Content
	got := m.recallFor(h, user, func(p string) bool { return p == "" || p == user || strings.Contains(user, p) })
	nids, out, n := m.packRecall(sys, kept, li, got, window, nil)
	if n == 0 {
		return ids, kept
	}
	logf("hes chat field  %d basins read  %d in the window", len(got), n)
	return nids, out
}

// withDroppedHES is the history the window could not hold: its turns are
// written into a field and read back for the last user turn.
func (m *Model) withDroppedHES(sys string, all, kept []Turn, window int) ([]int, []Turn) {
	base, err := m.render(sys, kept)
	if err != nil || len(kept) == 0 || len(kept) > len(all) {
		return base, kept
	}
	h, err := newChatField()
	if err != nil {
		return base, kept
	}
	var texts []string
	for _, t := range all[:len(all)-len(kept)] {
		if !strings.EqualFold(t.Role, "system") && strings.TrimSpace(t.Content) != "" {
			texts = append(texts, t.Content)
		}
	}
	for _, t := range texts {
		m.writeText(h, t)
	}
	li := lastUser(kept)
	if li < 0 {
		return base, kept
	}
	user := kept[li].Content
	got := m.recallFor(h, user, func(p string) bool { return p == "" || p == user })
	ids, out, n := m.packRecall(sys, kept, li, got, window, nil)
	if n == 0 {
		return base, kept
	}
	logf("hes dropped history  %d turns written  %d basins in the window", len(texts), n)
	return ids, out
}

// overflowTurn is a last user turn longer than the window. Its tail (where
// the question usually is) stays as the turn; the rest is written into a
// field, chunk by chunk, and read back for the tail in the document's own
// order, as much as the window takes. The window no longer bounds the input.
func (m *Model) overflowTurn(sys string, turns []Turn, window int) ([]Turn, error) {
	li := lastUser(turns)
	if li < 0 {
		return nil, fmt.Errorf("empty prompt")
	}
	text := turns[li].Content
	parts := m.chunks(text)
	if len(parts) < 2 {
		ids := m.pieces(text)
		keep := window / 4
		if len(ids) <= keep {
			return nil, ErrTooLong
		}
		parts = []string{m.tok.Decode(ids[:len(ids)-keep], true), m.tok.Decode(ids[len(ids)-keep:], true)}
	}
	// The tail: whole chunks from the end, up to a quarter of the window.
	tail, tailN := len(parts), 0
	for tail > 1 {
		n := len(m.pieces(parts[tail-1]))
		if tailN > 0 && tailN+n > window/4 {
			break
		}
		tail--
		tailN += n
	}
	head := parts[:tail]
	index := map[string]int{}
	for i, c := range head {
		index[c] = i
	}
	h, err := m.libraryFor(head)
	if err != nil {
		return nil, err
	}
	if chat := m.ensureHES(); chat != nil && hesFile != "" {
		for _, c := range head {
			m.writeKeyed(chat, c, m.pieces(c))
		}
	}
	m.logLaws("long input", h)
	m.logH1("long input", h)
	question := strings.Join(parts[tail:], "\n\n")
	kept := append([]Turn(nil), turns...)
	kept[li].Content = question
	// The read is keyed by what is asked: the turn's last paragraph (a long
	// paste ends with its question), not the whole kept tail.
	got := m.recallFor(h, lastParagraph(question), nil)
	docOrder := func(ps []string) []string {
		sort.SliceStable(ps, func(i, j int) bool { return index[ps[i]] < index[ps[j]] })
		return ps
	}
	_, out, n := m.packRecall(sys, kept, li, got, window, docOrder)
	logf("hes long input  %d tokens  %d chunks into the field  %d tokens kept as the turn  %d chunks read back",
		len(m.pieces(text)), len(head), tailN, n)
	return out, nil
}

// overflowUser is overflowTurn for one chat message: the head goes into the
// chat field (so later turns can read it too) and the tail stays.
func (m *Model) overflowUser(user string, reserve int) (string, bool) {
	h := m.ensureHES()
	if h == nil {
		return user, false
	}
	parts := m.chunks(user)
	window := m.ctxLen() - reserve
	if len(parts) < 2 {
		return user, false
	}
	tail, tailN := len(parts), 0
	for tail > 1 {
		n := len(m.pieces(parts[tail-1]))
		if tailN > 0 && tailN+n > window/4 {
			break
		}
		tail--
		tailN += n
	}
	for _, c := range parts[:tail] {
		if _, err := m.writeKeyed(h, c, m.pieces(c)); err != nil {
			logf("hes write  %v", err)
		}
	}
	logf("hes long message  %d chunks into the chat field  %d tokens kept", tail, tailN)
	return strings.Join(parts[tail:], "\n\n"), true
}

// noteHES writes one finished exchange into the chat field.
func (m *Model) noteHES(user, assistant string) {
	h := m.ensureHES()
	if h == nil {
		return
	}
	m.writeText(h, user)
	m.writeText(h, assistant)
	m.saveHES()
}

// earlierHES is user with the nearest forgotten basins of the chat field in
// front of it, as many as the tokens this turn still has free take.
func (m *Model) earlierHES(user string, reserve int) string {
	if chat == nil {
		return user
	}
	h := m.ensureHES()
	if h == nil || h.Writes() == 0 {
		return user
	}
	nTok := func(s string) int { return len(m.turnFor(s)) }
	budget := m.ctxLen() - reserve - chat.pos
	got := m.recallFor(h, user, func(p string) bool { return skipPayload(p, user) })
	var taken []string
	for _, r := range got {
		next := strings.Join(append(append([]string(nil), taken...), r.Payload), "\n\n")
		if nTok(hes.Prompt(user, next)) > budget {
			continue
		}
		taken = append(taken, r.Payload)
		logf("hes recall  basin %d  arrival %d  holonomy %s", r.Basin, r.Arrival, r.Hol)
	}
	if len(taken) == 0 {
		if len(got) == 0 {
			return user
		}
		fitted := hes.Fit(user, got[0].Payload, budget-nTok(user), nTok)
		if fitted == "" {
			return user
		}
		sightings = append(sightings, sighting{payload: got[0].Payload, fitted: fitted})
		return hes.Prompt(user, fitted)
	}
	joined := strings.Join(taken, "\n\n")
	for _, p := range taken {
		sightings = append(sightings, sighting{payload: p, fitted: joined})
	}
	return hes.Prompt(user, joined)
}

// --- a reply longer than the window ---------------------------------------

// maxSlides bounds how many times one reply moves past the window.
const maxSlides = 64

// lastRun is what begin used, for sliding the same reply on.
var lastRun struct {
	sys  string
	kept []Turn
	h    *hes.HES
}

// slide moves a reply that filled the window onto a fresh window: what was
// said so far goes into the field, and the new window holds the user turn
// with the basins the field returns for the reply's tail, then that tail.
func (m *Model) slide(said string, opt Opt) (*dialog, error) {
	sys, kept := lastRun.sys, append([]Turn(nil), lastRun.kept...)
	li := lastUser(kept)
	if li < 0 {
		return nil, fmt.Errorf("slide needs a user turn")
	}
	ids := m.pieces(said)
	keep := m.ctxLen() / 4
	if keep > len(ids) {
		keep = len(ids)
	}
	head, tail := m.tok.Decode(ids[:len(ids)-keep], true), ids[len(ids)-keep:]
	if lastRun.h == nil {
		h, err := newChatField()
		if err != nil {
			return nil, err
		}
		lastRun.h = h
	}
	m.writeText(lastRun.h, head)
	window := m.ctxLen() - room(opt) - len(tail)
	tailText := m.tok.Decode(tail, true)
	got := m.recallFor(lastRun.h, kept[li].Content+"\n"+tailText, nil)
	pre, out, _ := m.packRecall(sys, kept, li, got, window, nil)
	if pre == nil {
		var err error
		if pre, err = m.render(sys, out); err != nil {
			return nil, err
		}
	}
	if len(pre)+len(tail) > m.ctxLen()-room(opt) {
		return nil, fmt.Errorf("%w (the user turn alone fills the window)", ErrTooLong)
	}
	logf("hes slide  %d tokens said  %d into the field  %d kept  window %d", len(ids), len(ids)-keep, keep, len(pre)+len(tail))
	d := &dialog{}
	if err := m.prefill(d, append(pre, tail...)); err != nil {
		return nil, err
	}
	return d, nil
}

// transcriptHES generates past the window: each time the reply fills it, the
// reply slides onto a fresh window through the field.
func (m *Model) transcriptHES(d *dialog, opt Opt, emit func(string)) (string, error) {
	all, err := m.generate(d, opt, emit)
	for s := 0; err == nil && d.full && s < maxSlides; s++ {
		n := opt.N
		if n > 0 {
			n -= len(m.pieces(all))
			if n <= 0 {
				break
			}
		}
		next := opt
		next.N = n
		if opt.Halt != nil {
			prev := all
			next.Halt = func(s string) bool { return opt.Halt(prev + s) }
		}
		if d, err = m.slide(all, opt); err != nil {
			return all, err
		}
		var more string
		more, err = m.generate(d, next, emit)
		all += more
	}
	return all, err
}
