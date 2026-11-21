package tui

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"gptrun/internal/catalog"
	"gptrun/internal/generate"
)

func TestInputAtBottom(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loaded = true
	m.width = 80
	m.height = 24
	view := m.View()
	lines := strings.Split(view, "\n")
	if len(lines) < 20 {
		t.Fatalf("view has %d lines\n%s", len(lines), view)
	}
	top := strings.Join(lines[:8], "\n")
	bottom := strings.Join(lines[len(lines)-8:], "\n")
	if strings.Contains(top, "╰") || strings.Contains(top, "┘") {
		t.Fatalf("box starts at the top\n%s", view)
	}
	if !strings.Contains(bottom, "╰") && !strings.Contains(bottom, "╯") && !strings.Contains(bottom, "┘") {
		t.Fatalf("box is not at the bottom\n%s", view)
	}
	if !strings.Contains(view, "Ready") {
		t.Fatalf("empty chat has no ready state\n%s", view)
	}
}

func TestMouseWheel(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loaded = true
	m.width = 80
	m.height = 24
	up, _ := m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
	if up.(model).scroll != 0 {
		t.Fatalf("short transcript scrolled to %d", up.(model).scroll)
	}
	for i := 0; i < 40; i++ {
		m.turns = append(m.turns, turn{role: "user", text: strings.Repeat("a line of conversation ", 6)})
	}
	up, _ = m.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
	if up.(model).scroll != 3 {
		t.Fatalf("wheel up scroll = %d", up.(model).scroll)
	}
	down, _ := up.(model).Update(tea.MouseMsg{Button: tea.MouseButtonWheelDown})
	if down.(model).scroll != 0 {
		t.Fatalf("wheel down scroll = %d", down.(model).scroll)
	}
	flicked := up.(model)
	for i := 0; i < 80; i++ {
		next, _ := flicked.Update(tea.MouseMsg{Button: tea.MouseButtonWheelUp})
		flicked = next.(model)
	}
	if flicked.scroll != flicked.scrollLimit() {
		t.Fatalf("scroll %d, limit %d", flicked.scroll, flicked.scrollLimit())
	}
	back, _ := flicked.Update(tea.MouseMsg{Button: tea.MouseButtonWheelDown})
	if back.(model).scroll != flicked.scroll-3 {
		t.Fatalf("wheel down from the top stayed at %d", back.(model).scroll)
	}
}

func TestEscHintWhenIdle(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loaded = true
	m.width = 80
	m.height = 24
	m.escArmed = true
	view := m.View()
	if !strings.Contains(view, "Press Esc again to quit") {
		t.Fatalf("hint missing\n%s", view)
	}
}

func TestCtrlC(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loaded = true
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd == nil {
		t.Fatal("idle ctrl+c should quit")
	}
	m.busy = true
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	if cmd != nil {
		t.Fatal("busy ctrl+c should interrupt")
	}
	if next.(model).status != "interrupted" {
		t.Fatalf("status %q", next.(model).status)
	}
}

func TestLoadFailureShowsError(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loading = false
	m.loaded = false
	m.width = 80
	m.height = 24
	m.status = "open config: no such file"
	view := m.View()
	if !strings.Contains(view, "Could not load") || !strings.Contains(view, "no such file") {
		t.Fatalf("error not shown\n%s", view)
	}
	lines := strings.Split(view, "\n")
	bottom := strings.Join(lines[len(lines)-3:], "\n")
	if !strings.Contains(bottom, "Esc") {
		t.Fatalf("footer not pinned\n%s", view)
	}
}

func TestLinesFit(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge, Layers: 32}, generate.Opt{})
	m.width = 40
	m.height = 24
	m.loaded = true
	m.turns = []turn{{role: "user", text: strings.Repeat("word ", 40)}, {role: "assistant", text: strings.Repeat("answer ", 40), ask: "q", route: "stand"}}
	assertFits(t, m.View(), 40)
	m.loaded = false
	m.loading = true
	m.progress = 0.5
	m.status = "layer 12  q packed  down dense"
	assertFits(t, m.View(), 40)
}

func TestProgressMovesForward(t *testing.T) {
	steps := []string{
		"loading gemma",
		"decoding 10 tensors from a.safetensors",
		"decoded 10 tensors",
		"layer 0  q packed  down dense",
		"layer 3  q packed  down dense",
		"weights resident  4 layers  ctx 1",
		"uploading embedding  8 bytes",
		"layer 0 resident",
		"layer 3 resident",
		"ready gemma  4 layers  hidden 8",
	}
	prev := -1.0
	for _, s := range steps {
		f := progressOf(s, 4)
		if f < prev {
			t.Fatalf("%q went %.3f -> %.3f", s, prev, f)
		}
		prev = f
	}
	if prev != 1 {
		t.Fatalf("end %v", prev)
	}
	if progressOf("layer 0 resident", 4) <= progressOf("layer 3  q packed  down dense", 4) {
		t.Fatal("device pass rewound behind the cpu pass")
	}
	if progressOf("chat role", 4) != -1 {
		t.Fatal("chat is not a progress line")
	}
	if faceStatus("layer 2  q packed  down dense") != "Layer 3" {
		t.Fatal(faceStatus("layer 2  q packed  down dense"))
	}
}

func TestLoadProgressNeverRewinds(t *testing.T) {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge, Layers: 4}, generate.Opt{})
	var next tea.Model = m
	for _, s := range []string{"layer 3  q packed", "uploading embedding  8 bytes", "decoding again"} {
		next, _ = next.(model).Update(progressMsg{text: s})
	}
	if next.(model).progress < 0.6 {
		t.Fatalf("progress rewound to %v", next.(model).progress)
	}
}

func ratedModel() model {
	m := newModel(catalog.Model{Name: "hodge", Family: catalog.FamilyHodge}, generate.Opt{})
	m.loaded = true
	m.width = 80
	m.height = 24
	m.ask = "Hi"
	m.turns = []turn{
		{role: "user", text: "Hi"},
		{role: "assistant", text: "Hello there", ask: "Hi", route: "stand"},
	}
	return m
}

func TestRateRow(t *testing.T) {
	view := ratedModel().View()
	for _, s := range []string{"RHFL stand", "copy", "up", "down"} {
		if !strings.Contains(view, s) {
			t.Fatalf("missing %s\n%s", s, view)
		}
	}
}

func TestSpeedAfterReply(t *testing.T) {
	m := ratedModel()
	m.turns[1].speed = "42 tok/s"
	view := m.View()
	reply := strings.Index(view, "Hello there")
	rate := strings.Index(view, "42 tok/s")
	if reply < 0 || rate < 0 || rate < reply {
		t.Fatalf("speed should follow the reply\n%s", view)
	}
}

func TestCopyReply(t *testing.T) {
	var got string
	prev := writeClip
	writeClip = func(s string) error {
		got = s
		return nil
	}
	t.Cleanup(func() { writeClip = prev })
	m := ratedModel()
	_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'c'}, Alt: true})
	if cmd == nil {
		t.Fatal("alt+c")
	}
	msg := cmd()
	note, ok := msg.(infoMsg)
	if !ok || note.text != "Copied" || got != "Hello there" {
		t.Fatalf("msg %#v clip %q", msg, got)
	}
}

func TestThumbUp(t *testing.T) {
	m := ratedModel()
	_, hits := m.layout(m.pane())
	var up hit
	for _, h := range hits {
		if h.kind == hitUp {
			up = h
		}
	}
	if up.w == 0 {
		t.Fatal("no up control")
	}
	next, cmd := m.Update(tea.MouseMsg{
		X: up.col, Y: 2 + up.line,
		Button: tea.MouseButtonLeft, Action: tea.MouseActionPress,
	})
	if cmd != nil {
		t.Fatal(cmd)
	}
	got := next.(model)
	if got.turns[up.turn].vote != 1 || got.note != "RHFL up" {
		t.Fatalf("vote %d note %q", got.turns[up.turn].vote, got.note)
	}
}

func TestThumbDownLooksUp(t *testing.T) {
	m := ratedModel()
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}, Alt: true})
	if cmd == nil {
		t.Fatal("down")
	}
	got := next.(model)
	if got.turns[1].vote != -1 || got.turns[1].text != "Hello there" || !got.busy {
		t.Fatalf("%#v busy %v", got.turns[1], got.busy)
	}
}

func TestThumbDownRestores(t *testing.T) {
	m := ratedModel()
	m.turns[1].text = "Paris, from the notes"
	m.turns[1].prior = "I am not sure."
	m.turns[1].route = "wiki"
	next, cmd := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'d'}, Alt: true})
	if cmd == nil {
		t.Fatal("restore")
	}
	got := next.(model)
	if got.turns[1].text != "I am not sure." || got.turns[1].route != "" || got.turns[1].vote != -1 {
		t.Fatalf("%#v", got.turns[1])
	}
}

func TestPairsUseTheSpelling(t *testing.T) {
	got := pairs([]turn{
		{role: "user", text: "captial"},
		{role: "tool", label: "spell", text: "capital"},
		{role: "assistant", text: "Paris."},
	})
	if len(got) != 1 || got[0][0] != "capital" || got[0][1] != "Paris." {
		t.Fatalf("%#v", got)
	}
}

func assertFits(t *testing.T, view string, width int) {
	t.Helper()
	for _, line := range strings.Split(view, "\n") {
		if w := lipgloss.Width(line); w > width {
			t.Fatalf("line width %d\n%q", w, line)
		}
	}
}
