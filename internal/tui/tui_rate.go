package tui

import (
	"context"
	"strings"

	"github.com/atotto/clipboard"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"gptrun/internal/agent"
	"gptrun/internal/catalog"
	"gptrun/internal/generate"
	"gptrun/internal/hodge"
	"gptrun/internal/rhfl"
	"gptrun/internal/search"
)

type hit struct {
	line int
	col  int
	w    int
	kind string
	turn int
}

const (
	hitCopy = "copy"
	hitUp   = "up"
	hitDown = "down"
)

var writeClip = clipboard.WriteAll

func routeWord(route string) string {
	switch rhfl.Kind(route) {
	case rhfl.Wiki:
		return "wiki"
	case rhfl.Tool:
		return "tool"
	default:
		return "stand"
	}
}

func rateRow(t turn, turnIdx, width int) (string, []hit) {
	var b strings.Builder
	col := 2
	var hits []hit
	write := func(s string, style lipgloss.Style) {
		b.WriteString(style.Render(s))
		col += lipgloss.Width(s)
	}
	control := func(label, kind string, on bool) {
		if col >= width {
			return
		}
		gap := "  "
		b.WriteString(dimStyle.Render(gap))
		col += lipgloss.Width(gap)
		if col >= width {
			return
		}
		style := dimStyle
		if on {
			style = accent
		}
		w := lipgloss.Width(label)
		if col+w > width {
			w = width - col
		}
		if w > 0 {
			hits = append(hits, hit{col: col, w: w, kind: kind, turn: turnIdx})
		}
		write(label, style)
	}
	if t.speed != "" {
		write(t.speed, dimStyle)
		gap := "  "
		b.WriteString(dimStyle.Render(gap))
		col += lipgloss.Width(gap)
	}
	write("RHFL "+routeWord(t.route), dimStyle)
	control("copy", hitCopy, false)
	control("up", hitUp, t.vote > 0)
	control("down", hitDown, t.vote < 0)
	line := gutter.Render("│ ") + b.String()
	if lipgloss.Width(line) > width {
		line = clip(line, width)
	}
	return line, hits
}

func (m model) controlAt(x, y int) (hit, bool) {
	if m.loading || !m.loaded {
		return hit{}, false
	}
	width := m.pane()
	_, viewH := m.chatBox(width)
	lines, hits := m.layout(width)
	rel := y - 2
	if rel < 0 || rel >= viewH {
		return hit{}, false
	}
	scroll := m.scroll
	if limit := len(lines) - viewH; scroll > limit {
		scroll = max(0, limit)
	}
	end := len(lines) - scroll
	start := end - viewH
	if start < 0 {
		start = 0
	}
	idx := start + rel
	for _, h := range hits {
		if h.line == idx && x >= h.col && x < h.col+h.w {
			return h, true
		}
	}
	return hit{}, false
}

func (m model) lastAssistant() int {
	for i := len(m.turns) - 1; i >= 0; i-- {
		if m.turns[i].role == "assistant" {
			return i
		}
	}
	return -1
}

func (m model) lastOutput() string {
	if i := m.lastAssistant(); i >= 0 {
		return m.turns[i].text
	}
	return ""
}

func (m model) lastRated() int {
	for i := len(m.turns) - 1; i >= 0; i-- {
		t := m.turns[i]
		if t.role == "assistant" && strings.TrimSpace(t.text) != "" && (t.ask != "" || t.route != "") {
			return i
		}
	}
	return -1
}

func (m model) markReply(kind, prior string) model {
	i := m.lastAssistant()
	if i < 0 {
		return m
	}
	if kind != "" {
		m.turns[i].route = kind
	}
	if prior != "" && m.turns[i].prior == "" {
		m.turns[i].prior = prior
	}
	if m.turns[i].ask == "" {
		m.turns[i].ask = m.ask
	}
	return m
}

func (m model) finishReply(speed string) model {
	i := m.lastAssistant()
	if i < 0 || strings.TrimSpace(m.turns[i].text) == "" {
		return m
	}
	if m.turns[i].ask == "" || m.turns[i].ask != m.ask {
		return m
	}
	if speed != "" {
		m.turns[i].speed = speed
	}
	if m.turns[i].route == "" {
		m.turns[i].route = string(rhfl.Stand)
	}
	if m.thought != "" && m.turns[i].thought == "" {
		m.turns[i].thought = m.thought
	}
	m.thought = ""
	return m
}

func (m model) use(h hit) (model, tea.Cmd) {
	if h.turn < 0 || h.turn >= len(m.turns) {
		return m, nil
	}
	switch h.kind {
	case hitCopy:
		return m, copyCmd(m.turns[h.turn].text)
	case hitUp:
		if m.busy || m.loading {
			return m, nil
		}
		m.turns[h.turn].vote = 1
		m.note = "RHFL up"
		return m, nil
	case hitDown:
		if m.busy || m.loading {
			return m, nil
		}
		return m.voteDown(h.turn)
	default:
		return m, nil
	}
}

func (m model) voteDown(i int) (model, tea.Cmd) {
	if m.turns[i].vote < 0 {
		return m, nil
	}
	m.turns[i].vote = -1
	t := m.turns[i]
	m.note = "RHFL down"
	cur, ok := m.current()
	if !ok || cur.Family != catalog.FamilyHodge || i != m.lastRated() {
		return m, nil
	}
	if t.route == string(rhfl.Wiki) && strings.TrimSpace(t.prior) != "" {
		m.turns[i].text = t.prior
		m.turns[i].prior = ""
		m.turns[i].route = string(rhfl.Stand)
		m.busy = true
		m.status = "Restoring the reply"
		return m, restoreCmd(cur.Name, pairs(m.turns))
	}
	if t.route == string(rhfl.Tool) || strings.TrimSpace(t.ask) == "" {
		return m, nil
	}
	m.busy = true
	m.status = "Looking it up"
	return m, correctCmd(cur.Name, i, t.ask, t.text, m.opt)
}

func copyCmd(text string) tea.Cmd {
	return func() tea.Msg {
		text = strings.TrimSpace(text)
		if text == "" {
			return infoMsg{text: "Nothing to copy"}
		}
		if err := writeClip(text); err != nil {
			return infoMsg{text: "Copy failed"}
		}
		return infoMsg{text: "Copied"}
	}
}

func correctCmd(name string, turn int, question, prior string, opt generate.Opt) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 8)
		go func() {
			defer close(ch)
			hodge.ClearInterrupt()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			agent.Arm(cancel)
			defer agent.Arm(nil)
			prompt, sources, gerr := search.New().Ground(ctx, question, func(s string) {
				ch <- statusMsg{text: s}
			}, nil)
			if hodge.Interrupted() || ctx.Err() != nil {
				ch <- voteMsg{turn: turn, err: hodge.ErrInterrupted}
				return
			}
			corrected := ""
			var err error
			if gerr == nil && strings.TrimSpace(prompt) != "" {
				corrected, err = hodge.Continue(name, prompt, generate.Greedy(opt), nil)
				if err == nil {
					corrected = search.Settle(question, prompt, corrected)
				}
			}
			if s := strings.TrimSpace(sources); s != "" && corrected != "" {
				corrected += "\n\n" + s
			}
			ch <- voteMsg{
				turn: turn, text: strings.TrimSpace(corrected), prior: prior,
				route: string(rhfl.Wiki), err: err,
			}
		}()
		return ch
	}
}

func restoreCmd(name string, seq [][2]string) tea.Cmd {
	return func() tea.Msg {
		hodge.Reset(name)
		for _, p := range seq {
			if err := hodge.Absorb(name, p[0], p[1]); err != nil && err != hodge.ErrInterrupted {
				return voteMsg{err: err, restore: true}
			}
		}
		return voteMsg{restore: true}
	}
}

func pairs(turns []turn) [][2]string {
	var out [][2]string
	var user string
	var notes []string
	have := false
	for _, t := range turns {
		switch t.role {
		case "user":
			user = t.text
			notes = nil
			have = true
		case "tool":
			if t.label == "spell" {
				user = t.text
				have = true
				continue
			}
			if strings.TrimSpace(t.text) != "" {
				notes = append(notes, t.text)
			}
		case "assistant":
			if !have && len(notes) == 0 {
				continue
			}
			if len(notes) > 0 {
				extra := strings.Join(notes, "\n\n")
				if user != "" {
					user += "\n\n" + extra
				} else {
					user = extra
				}
				notes = nil
				have = true
			}
			if strings.TrimSpace(t.text) == "" {
				have = false
				user = ""
				continue
			}
			out = append(out, [2]string{user, t.text})
			have = false
			user = ""
		}
	}
	return out
}
