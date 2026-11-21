// Package tui is the Bubble Tea face of ModelRun.
package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"gptrun/internal/agent"
	"gptrun/internal/catalog"
	"gptrun/internal/generate"
	"gptrun/internal/hodge"
	"gptrun/internal/rhfl"
	"gptrun/internal/search"
	"gptrun/internal/spell"
	"gptrun/internal/version"
	"gptrun/internal/vm"
)

var (
	titleStyle = lipgloss.NewStyle().Bold(true)
	dimStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	accent     = lipgloss.NewStyle().Foreground(lipgloss.Color("74"))
	bad        = lipgloss.NewStyle().Foreground(lipgloss.Color("203"))
	youStyle   = lipgloss.NewStyle().Bold(true)
	gutter     = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	toolStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("109"))
	boxStyle   = lipgloss.NewStyle().
			Border(lipgloss.RoundedBorder()).
			BorderForeground(lipgloss.Color("240")).
			Padding(0, 1)
	boxBusy = lipgloss.NewStyle().
		Border(lipgloss.RoundedBorder()).
		BorderForeground(lipgloss.Color("238")).
		Padding(0, 1)
)

type turn struct {
	role    string
	text    string
	label   string
	ask     string // question this reply answers
	prior   string // model words before an RHFL replacement
	route   string // rhfl kind: stand, wiki, or tool
	vote    int    // 1 up, -1 down
	speed   string // decode rate of this reply, such as "42 tok/s"
	thought string // chain of thought for this reply, shown once
}

type replyMsg struct {
	text string
	err  error
}

type tokenMsg struct {
	text  string
	done  bool
	err   error
	speed string
}

type statusMsg struct{ text string }

type progressMsg struct{ text string }

type loadedMsg struct{ err error }

type infoMsg struct{ text string }

type toolMsg struct {
	label string
	text  string
}

type replaceMsg struct {
	text  string
	prior string
	route string
}

type routeMsg struct {
	kind  string
	prior string
}

type voteMsg struct {
	turn    int
	text    string
	prior   string
	route   string
	restore bool
	err     error
}

type model struct {
	cur      catalog.Model
	turns    []turn
	skill    string
	input    textarea.Model
	spin     spinner.Model
	busy     bool
	width    int
	height   int
	status   string
	opt      generate.Opt
	stream   chan tea.Msg
	scroll   int
	loading  bool
	loaded   bool
	progress float64
	escArmed bool
	ask      string
	note     string
	thought  string // thought streaming before the answer
}

// RunFile starts chat for the checkpoint named by a config file.
func RunFile(path string, opt generate.Opt) error {
	meta, err := catalog.Load(path)
	if err != nil {
		return err
	}
	m := newModel(meta, opt)
	m.loading = true
	m.input.Focus()
	_, err = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion()).Run()
	return err
}

// Run starts chat for one checkpoint. The name comes from the CLI.
func Run(name string, opt generate.Opt) error {
	if name == "" {
		return fmt.Errorf("gptrun run <model>")
	}
	meta, ok, err := catalog.Get(name)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no model %q (gptrun list)", name)
	}
	m := newModel(meta, opt)
	if meta.Family == catalog.FamilyHodge {
		m.loading = true
	} else {
		m.loaded = true
	}
	m.input.Focus()
	_, err = tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion()).Run()
	return err
}

func newModel(cur catalog.Model, opt generate.Opt) model {
	ta := textarea.New()
	ta.Placeholder = "Ask a question"
	ta.CharLimit = 0
	ta.SetHeight(3)
	ta.ShowLineNumbers = false
	ta.Prompt = "› "
	ta.FocusedStyle.Prompt = accent
	ta.FocusedStyle.Placeholder = dimStyle
	ta.BlurredStyle.Placeholder = dimStyle
	ta.KeyMap.InsertNewline.SetKeys("shift+enter", "alt+enter")
	sp := spinner.New()
	sp.Spinner = spinner.Dot
	sp.Style = accent
	note := ""
	if msg, stale := version.Stale(); stale {
		note = "Warning: " + msg
	}
	return model{
		cur:    cur,
		input:  ta,
		spin:   sp,
		width:  80,
		height: 24,
		opt:    opt,
		note:   note,
	}
}

func (m model) Init() tea.Cmd {
	cmds := []tea.Cmd{textarea.Blink, m.spin.Tick}
	if m.loading {
		cmds = append(cmds, loadCmd(m.cur.Name))
	}
	return tea.Batch(cmds...)
}

func (m model) current() (catalog.Model, bool) {
	if m.cur.Name == "" {
		return catalog.Model{}, false
	}
	return m.cur, true
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.input.SetWidth(msg.Width - 4)
		return m, nil
	case infoMsg:
		m.note = msg.text
		return m, nil
	case spinner.TickMsg:
		var cmd tea.Cmd
		m.spin, cmd = m.spin.Update(msg)
		return m, cmd
	case replyMsg:
		m.busy = false
		if msg.err != nil {
			m.turns = append(m.turns, turn{role: "error", text: msg.err.Error()})
		} else {
			m.turns = append(m.turns, turn{role: "assistant", text: msg.text, ask: m.ask, route: string(rhfl.Stand)})
		}
		return m, nil
	case chan tea.Msg:
		m.stream = msg
		return m, waitStream(msg)
	case progressMsg:
		if f := progressOf(msg.text, m.cur.Layers); f > m.progress {
			m.progress = f
		}
		m.status = msg.text
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case loadedMsg:
		m.loading = false
		m.stream = nil
		if msg.err != nil {
			m.turns = append(m.turns, turn{role: "error", text: msg.err.Error()})
			m.status = msg.err.Error()
			return m, nil
		}
		m.loaded = true
		m.progress = 1
		m.status = "ready"
		return m, nil
	case statusMsg:
		m.status = msg.text
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case tokenMsg:
		if msg.done {
			m.busy = false
			m.stream = nil
			m = m.finishReply(msg.speed)
			if errors.Is(msg.err, hodge.ErrInterrupted) {
				m.status = "interrupted"
			} else if msg.err != nil {
				m.turns = append(m.turns, turn{role: "error", text: msg.err.Error()})
			}
			return m, nil
		}
		m = m.appendPiece(msg.text)
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case tea.MouseMsg:
		m.escArmed = false
		switch msg.Button {
		case tea.MouseButtonWheelUp:
			m.scroll = m.bound(m.bound(m.scroll) + 3)
		case tea.MouseButtonWheelDown:
			m.scroll = m.bound(m.bound(m.scroll) - 3)
		case tea.MouseButtonLeft:
			if msg.Action != tea.MouseActionPress {
				return m, nil
			}
			if h, ok := m.controlAt(msg.X, msg.Y); ok {
				return m.use(h)
			}
		}
		return m, nil
	case routeMsg:
		m = m.markReply(msg.kind, msg.prior)
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case voteMsg:
		m.busy = false
		m.stream = nil
		if errors.Is(msg.err, hodge.ErrInterrupted) {
			m.status = "interrupted"
			return m, nil
		}
		if msg.err != nil {
			m.note = oneLine(msg.err.Error())
			return m, nil
		}
		if !msg.restore && msg.text != "" && msg.turn >= 0 && msg.turn < len(m.turns) {
			m.turns[msg.turn].prior = msg.prior
			m.turns[msg.turn].text = msg.text
			m.turns[msg.turn].route = msg.route
		}
		m.note = "RHFL down"
		return m, nil
	case replaceMsg:
		if n := len(m.turns); n > 0 && m.turns[n-1].role == "assistant" {
			if strings.TrimSpace(msg.text) == "" {
				m.turns = m.turns[:n-1]
			} else {
				if msg.prior != "" {
					m.turns[n-1].prior = msg.prior
				}
				if msg.route != "" {
					m.turns[n-1].route = msg.route
				}
				if m.turns[n-1].ask == "" {
					m.turns[n-1].ask = m.ask
				}
				m.turns[n-1].text = msg.text
			}
		} else if strings.TrimSpace(msg.text) != "" {
			m.turns = append(m.turns, turn{
				role: "assistant", text: msg.text, ask: m.ask, prior: msg.prior, route: msg.route,
			})
		}
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case toolMsg:
		label := msg.label
		if label == "" {
			label = "sandbox"
		}
		next := turn{role: "tool", label: label, text: msg.text}
		if n := len(m.turns); n > 0 && m.turns[n-1].role == "assistant" && agent.Parse(m.turns[n-1].text) != nil {
			m.turns[n-1] = next
		} else {
			m.turns = append(m.turns, next)
		}
		m.scroll = 0
		if m.stream != nil {
			return m, waitStream(m.stream)
		}
		return m, nil
	case tea.KeyMsg:
		if msg.String() != "esc" {
			m.escArmed = false
		}
		switch msg.String() {
		case "ctrl+c":
			if m.busy {
				if m.status == "interrupted" {
					return m, tea.Quit
				}
				hodge.Interrupt()
				agent.Cancel()
				m.status = "interrupted"
				return m, nil
			}
			return m, tea.Quit
		case "esc":
			if m.escArmed {
				return m, tea.Quit
			}
			m.escArmed = true
			m.status = "Press Esc again to quit"
			return m, nil
		case "alt+c":
			return m, copyCmd(m.lastOutput())
		case "alt+u":
			if i := m.lastRated(); i >= 0 {
				return m.use(hit{kind: hitUp, turn: i})
			}
			return m, nil
		case "alt+d":
			if i := m.lastRated(); i >= 0 {
				return m.use(hit{kind: hitDown, turn: i})
			}
			return m, nil
		case "pgup", "ctrl+u":
			m.scroll += m.page()
			return m, nil
		case "pgdown", "ctrl+d":
			if m.scroll > 0 {
				m.scroll -= m.page()
				if m.scroll < 0 {
					m.scroll = 0
				}
				return m, nil
			}
		case "enter":
			if m.busy || m.loading || !m.loaded {
				return m, nil
			}
			text := strings.TrimSpace(m.input.Value())
			if text == "" {
				return m, nil
			}
			cur, ok := m.current()
			if !ok {
				return m, nil
			}
			m.input.Reset()
			m.note = ""
			m.turns = append(m.turns, turn{role: "user", text: text})
			if fixed, ok := spell.Adapt(cur.Tokenizer, text); ok {
				m.turns = append(m.turns, turn{role: "tool", text: fixed, label: "spell"})
				text = fixed
			}
			m.scroll = 0
			if name, arg, ok := agent.Verb(text); ok {
				var reply string
				var err error
				m, reply, err = m.verb(name, arg)
				if err != nil {
					m.turns = append(m.turns, turn{role: "error", text: err.Error()})
				} else if reply != "" {
					m.turns = append(m.turns, turn{role: "assistant", text: reply})
				}
				return m, nil
			}
			m.ask = text
			m.busy = true
			if call := agent.Slash(text); call != nil {
				m.status = "sandbox"
				return m, execCmd(call.Cmd())
			}
			if cur.Family == catalog.FamilyHodge {
				m.thought = ""
				return m, streamCmd(cur.Name, text, m.opt)
			}
			return m, func() tea.Msg {
				rep, err := generate.Run(cur.Name, text)
				if err != nil {
					return replyMsg{err: err}
				}
				return replyMsg{text: rep.Text}
			}
		}
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m model) appendPiece(s string) model {
	if s == "" {
		return m
	}
	if n := len(m.turns); n > 0 && m.turns[n-1].role == "assistant" {
		m.turns[n-1].text += s
		return m
	}
	m.turns = append(m.turns, turn{role: "assistant", text: s, ask: m.ask, thought: m.thought})
	return m
}

func loadCmd(name string) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 64)
		go func() {
			defer close(ch)
			hodge.SetLog(func(s string) {
				ch <- progressMsg{text: s}
			})
			err := hodge.Prepare(name)
			hodge.SetLog(nil)
			ch <- loadedMsg{err: err}
			if err == nil {
				// The chat's system prompt is read while the person types. A
				// turn sent meanwhile waits for this and then skips that reading.
				_ = hodge.Warm(name)
			}
		}()
		return ch
	}
}

// progressOf maps a load line onto 0..1. CPU layers and device layers share
// the words "layer N", so the device pass is its own band. Unknown lines
// return -1 and the caller keeps the last value, which stops the bar rewinding.
func progressOf(s string, layers int) float64 {
	if layers < 1 {
		layers = 32
	}
	switch {
	case strings.HasPrefix(s, "loading"):
		return 0.02
	case strings.HasPrefix(s, "decoding"), strings.HasPrefix(s, "reading"):
		return 0.06
	case strings.HasPrefix(s, "decoded"), strings.HasPrefix(s, "read "):
		return 0.14
	case strings.HasPrefix(s, "weights"):
		return 0.58
	case strings.HasPrefix(s, "uploading"):
		return 0.64
	case strings.HasPrefix(s, "ready"):
		return 1
	}
	var n int
	if _, err := fmt.Sscanf(s, "layer %d resident", &n); err == nil {
		return band(0.66, 0.96, n, layers)
	}
	if _, err := fmt.Sscanf(s, "layer %d", &n); err == nil {
		return band(0.16, 0.54, n, layers)
	}
	return -1
}

func band(lo, hi float64, n, layers int) float64 {
	if n < 0 {
		n = 0
	}
	if n >= layers {
		n = layers - 1
	}
	return lo + (hi-lo)*float64(n+1)/float64(layers)
}

// faceStatus is the load line a person reads. The raw line still drives the bar.
func faceStatus(s string) string {
	s = oneLine(s)
	if s == "" {
		return ""
	}
	var n int
	if _, err := fmt.Sscanf(s, "layer %d resident", &n); err == nil {
		return fmt.Sprintf("Layer %d on device", n+1)
	}
	if _, err := fmt.Sscanf(s, "layer %d", &n); err == nil {
		return fmt.Sprintf("Layer %d", n+1)
	}
	switch {
	case strings.HasPrefix(s, "decoding"), strings.HasPrefix(s, "reading"):
		return "Reading the checkpoint"
	case strings.HasPrefix(s, "decoded"), strings.HasPrefix(s, "read "):
		return "Checkpoint read"
	case strings.HasPrefix(s, "uploading"):
		return "Uploading weights"
	case strings.HasPrefix(s, "weights"):
		return "Weights on device"
	case strings.HasPrefix(s, "loading"):
		return "Opening the checkpoint"
	case strings.HasPrefix(s, "ready"):
		return "Ready"
	case strings.HasPrefix(s, "prefill"), strings.HasPrefix(s, "prompt"):
		return "Reading the prompt"
	case strings.HasPrefix(s, "generated"):
		return "Done"
	case strings.HasPrefix(s, "chat trimmed"), strings.HasPrefix(s, "history trimmed"):
		return "Trimming older turns to fit the context"
	case strings.HasPrefix(s, "GPU stopped"):
		return "The GPU stopped; reloading the model"
	case s == "model reloaded":
		return "Model reloaded"
	case strings.HasPrefix(s, "chat "):
		return "Preparing chat"
	default:
		return s
	}
}

func streamCmd(name, text string, opt generate.Opt) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 32)
		go func() {
			defer close(ch)
			hodge.ClearInterrupt()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			agent.Arm(cancel)
			defer agent.Arm(nil)
			hodge.SetLog(func(s string) {
				select {
				case ch <- statusMsg{text: s}:
				default:
				}
			})
			defer hodge.SetLog(nil)
			ask := func(user string) (string, error) {
				var b strings.Builder
				_, err := hodge.Continue(name, user, opt, func(s string) {
					b.WriteString(s)
					ch <- tokenMsg{text: s}
				})
				return b.String(), err
			}
			quiet := func(user string) (string, error) {
				return hodge.Continue(name, user, opt, nil)
			}
			runTools := func(reply string) (bool, error) {
				box := vm.New()
				ran := false
				var err error
				for round := 0; round < agent.MaxRounds && err == nil && !hodge.Interrupted(); round++ {
					call := agent.CallIn(reply)
					if call == nil {
						break
					}
					if lead := agent.WithoutCall(reply); lead != "" && !ran {
						ch <- replaceMsg{text: lead}
					}
					ch <- statusMsg{text: call.Tool + "…"}
					label, out, ok := runTool(ctx, box, call)
					if !ok {
						break
					}
					ran = true
					if hodge.Interrupted() || ctx.Err() != nil {
						ch <- toolMsg{label: label, text: "Interrupted."}
						err = hodge.ErrInterrupted
						break
					}
					ch <- toolMsg{label: label, text: out}
					next := agent.Wire(out)
					if call.Tool != "exec" {
						next = agent.LookupWire(out)
					}
					reply, err = quiet(next)
				}
				if ran && err == nil && !hodge.Interrupted() && strings.TrimSpace(reply) != "" {
					if agent.CallIn(reply) == nil {
						ch <- tokenMsg{text: agent.WithoutCall(reply)}
					} else if again, aerr := ask("Answer the person in plain words. Do not use JSON."); aerr != nil {
						err = aerr
					} else if strings.TrimSpace(again) != "" {
						ch <- tokenMsg{text: again}
					}
				}
				if ran {
					ch <- routeMsg{kind: string(rhfl.Tool)}
				}
				return ran, err
			}
			// The reply is the model's answer to the turn, streamed as it is
			// written. No thought or route pass runs first.
			reply, err := ask(text)
			speed := hodge.SpeedLine()
			if err != nil || hodge.Interrupted() || ctx.Err() != nil {
				if err == nil {
					err = hodge.ErrInterrupted
				}
				ch <- tokenMsg{done: true, err: err, speed: speed}
				return
			}
			kind, _ := rhfl.Of(text, reply)
			switch kind {
			case rhfl.Tool:
				ran, rerr := runTools(reply)
				err = rerr
				if ran {
					speed = hodge.SpeedLine()
				}
			case rhfl.Wiki:
				prompt, sources, gerr := search.New().Ground(ctx, text, func(s string) {
					ch <- statusMsg{text: s}
				}, nil)
				if hodge.Interrupted() || ctx.Err() != nil {
					ch <- tokenMsg{done: true, err: hodge.ErrInterrupted, speed: speed}
					return
				}
				corrected := ""
				if gerr == nil && strings.TrimSpace(prompt) != "" {
					corrected, err = hodge.Continue(name, prompt, generate.Greedy(opt), nil)
					if err == nil {
						corrected = search.Settle(text, prompt, corrected)
					}
				}
				if s := strings.TrimSpace(sources); s != "" && corrected != "" {
					corrected += "\n\n" + s
				}
				if strings.TrimSpace(corrected) != "" {
					speed = hodge.SpeedLine()
					ch <- replaceMsg{text: corrected, prior: reply, route: string(rhfl.Wiki)}
				}
			}
			ch <- tokenMsg{done: true, err: err, speed: speed}
		}()
		return ch
	}
}

func execCmd(command string) tea.Cmd {
	return func() tea.Msg {
		ch := make(chan tea.Msg, 4)
		go func() {
			defer close(ch)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			agent.Arm(cancel)
			defer agent.Arm(nil)
			hodge.ClearInterrupt()
			out, _ := vm.New().Exec(ctx, command)
			if hodge.Interrupted() || ctx.Err() != nil {
				ch <- toolMsg{text: "$ " + command + "\nInterrupted."}
				ch <- tokenMsg{done: true, err: hodge.ErrInterrupted}
				return
			}
			ch <- toolMsg{text: "$ " + command + "\n" + out}
			ch <- tokenMsg{done: true}
		}()
		return ch
	}
}

func waitStream(ch chan tea.Msg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return tokenMsg{done: true}
		}
		return msg
	}
}

func (m model) View() string {
	width := m.width
	if width < 20 {
		width = 20
	}
	left := titleStyle.Render("gptrun")
	roomR := width - lipgloss.Width(left) - 1
	if roomR < 1 {
		roomR = 1
	}
	right := dimStyle.Render(clip(m.meta(), roomR))
	gap := width - lipgloss.Width(left) - lipgloss.Width(right)
	if gap < 1 {
		gap = 1
	}
	head := left + strings.Repeat(" ", gap) + right
	rule := dimStyle.Render(strings.Repeat("─", width))
	foot := m.footer()
	footH := strings.Count(foot, "\n") + 1
	room := m.height - 2 - footH
	if room < 8 {
		room = 8
	}
	return head + "\n" + rule + "\n" + m.viewChat(room) + "\n" + foot
}

func (m model) meta() string {
	cur, ok := m.current()
	if !ok {
		return ""
	}
	name := cur.Detail
	if name == "" {
		name = cur.Name
	}
	var parts []string
	if name != "" {
		parts = append(parts, name)
	}
	if cur.Family != "" && cur.Family != name {
		parts = append(parts, cur.Family)
	}
	if cur.Layers > 0 {
		parts = append(parts, fmt.Sprintf("%d layers", cur.Layers))
	}
	return strings.Join(parts, "  ·  ")
}

func (m model) speaker() string {
	cur, ok := m.current()
	if !ok {
		return "Hodge"
	}
	if cur.Detail != "" {
		return cur.Detail
	}
	if cur.Name != "" {
		return cur.Name
	}
	return "Hodge"
}

func (m model) footer() string {
	width := m.width
	if width < 20 {
		width = 20
	}
	keys := "Enter send · Alt+C copy · Alt+U up · Alt+D down · "
	if m.busy || m.loading {
		keys += "Ctrl+C stop · Esc twice quit"
	} else {
		keys += "Ctrl+C quit · Esc twice quit"
	}
	return m.stateLine(width) + "\n" + dimStyle.Render(clip(keys, width))
}

func (m model) stateLine(width int) string {
	if m.escArmed {
		return accent.Render(clip("Press Esc again to quit", width))
	}
	if m.loading || m.busy {
		s := faceStatus(m.status)
		if s == "" {
			if m.loading {
				s = "Loading"
			} else {
				s = "Working"
			}
		}
		mark := m.spin.View()
		room := width - lipgloss.Width(mark) - 1
		if room < 1 {
			room = 1
		}
		return mark + " " + dimStyle.Render(clip(s, room))
	}
	if !m.loaded {
		return bad.Render(clip("Not loaded", width))
	}
	if m.note != "" {
		return dimStyle.Render(clip(m.note, width))
	}
	if m.skill != "" {
		return dimStyle.Render(clip("Skill  "+m.skill, width))
	}
	return dimStyle.Render(clip("Ready  "+version.Short(), width))
}

func (m model) bound(scroll int) int {
	if scroll < 0 {
		return 0
	}
	if limit := m.scrollLimit(); scroll > limit {
		return limit
	}
	return scroll
}

func (m model) scrollLimit() int {
	width := m.width
	if width < 20 {
		width = 20
	}
	room := m.height - 4
	if room < 8 {
		room = 8
	}
	if m.loading || !m.loaded {
		return 0
	}
	m.input.SetWidth(width - 6)
	frame := boxStyle
	if m.busy {
		frame = boxBusy
	}
	box := frame.Width(width - 4).Render(m.input.View())
	boxH := strings.Count(box, "\n") + 1
	viewH := room - boxH - 1
	if viewH < 1 {
		viewH = 1
	}
	limit := len(m.transcript(width)) - viewH
	if limit < 0 {
		return 0
	}
	return limit
}

func (m model) pane() int {
	if m.width < 20 {
		return 20
	}
	return m.width
}

func (m model) chatRoom() int {
	footH := strings.Count(m.footer(), "\n") + 1
	room := m.height - 2 - footH
	if room < 8 {
		room = 8
	}
	return room
}

func (m model) chatBox(width int) (string, int) {
	m.input.SetWidth(width - 6)
	frame := boxStyle
	if m.busy {
		frame = boxBusy
	}
	box := frame.Width(width - 4).Render(m.input.View())
	boxH := strings.Count(box, "\n") + 1
	viewH := m.chatRoom() - boxH - 1
	if viewH < 1 {
		viewH = 1
	}
	return box, viewH
}

func (m model) viewChat(room int) string {
	width := m.pane()
	if m.loading || !m.loaded {
		return m.loadingView(width, room, m.speaker())
	}
	box, viewH := m.chatBox(width)
	lines := m.transcript(width)
	scroll := m.scroll
	if limit := len(lines) - viewH; scroll > limit {
		scroll = max(0, limit)
	}
	end := len(lines) - scroll
	start := end - viewH
	if start < 0 {
		start = 0
	}
	if end < start {
		end = start
	}
	if end > len(lines) {
		end = len(lines)
	}
	var shown []string
	if start < end {
		shown = lines[start:end]
	}
	for len(shown) < viewH {
		shown = append(shown, "")
	}
	return strings.Join(shown, "\n") + "\n\n" + box
}

func (m model) loadingView(width, room int, name string) string {
	if room < 1 {
		room = 1
	}
	var lines []string
	lines = append(lines, "")
	if m.loading {
		mark := m.spin.View()
		roomN := width - lipgloss.Width(mark) - 2
		if roomN < 4 {
			roomN = 4
		}
		lines = append(lines, mark+"  "+titleStyle.Render(clip("Loading "+name, roomN)))
		lines = append(lines, "")
		lines = append(lines, m.barLine(width))
		lines = append(lines, "")
	} else {
		lines = append(lines, bad.Render(clip("Could not load "+name, width)))
		lines = append(lines, "")
	}
	label := faceStatus(m.status)
	if label == "" && m.loading {
		label = "Starting"
	}
	style := dimStyle
	if !m.loading {
		style = bad
	}
	for _, line := range wrapText(label, width) {
		lines = append(lines, style.Render(line))
	}
	for len(lines) < room {
		lines = append(lines, "")
	}
	if len(lines) > room {
		lines = lines[:room]
	}
	return strings.Join(lines, "\n")
}

func (m model) barLine(width int) string {
	pct := fmt.Sprintf("%3d%%", int(m.progress*100+0.5))
	extra := 1 + len(pct)
	inner := width - extra
	if inner < 1 {
		inner = 1
	}
	filled := int(m.progress*float64(inner) + 0.5)
	if filled < 0 {
		filled = 0
	}
	if filled > inner {
		filled = inner
	}
	bar := accent.Render(strings.Repeat("█", filled)) + dimStyle.Render(strings.Repeat("░", inner-filled))
	return bar + " " + dimStyle.Render(pct)
}

func (m model) transcript(width int) []string {
	lines, _ := m.layout(width)
	return lines
}

func (m model) layout(width int) ([]string, []hit) {
	wrapAt := width - 2
	if wrapAt < 8 {
		wrapAt = 8
	}
	if len(m.turns) == 0 {
		return m.emptyTranscript(wrapAt), nil
	}
	name := m.speaker()
	var lines []string
	var hits []hit
	last := len(m.turns) - 1
	for i, t := range m.turns {
		live := m.busy && i == last && t.role == "assistant"
		switch t.role {
		case "user":
			lines = append(lines, say(youStyle, "You", t.text, width, wrapAt, false)...)
		case "error":
			lines = append(lines, say(bad, "Error", t.text, width, wrapAt, false)...)
		case "tool":
			label := t.label
			if label == "" {
				label = "sandbox"
			}
			lines = append(lines, say(toolStyle, label, t.text, width, wrapAt, false)...)
		default:
			if t.thought != "" {
				lines = append(lines, say(dimStyle, "Thought", t.thought, width, wrapAt, false)...)
			}
			lines = append(lines, say(accent, name, t.text, width, wrapAt, live)...)
			if !live && strings.TrimSpace(t.text) != "" && (t.ask != "" || t.route != "") {
				row, hs := rateRow(t, i, width)
				for n := range hs {
					hs[n].line = len(lines)
				}
				lines = append(lines, row)
				hits = append(hits, hs...)
			}
		}
		lines = append(lines, "")
	}
	if m.thought != "" && (len(m.turns) == 0 || m.turns[last].thought == "") {
		lines = append(lines, say(dimStyle, "Thought", m.thought, width, wrapAt, true)...)
	}
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines, hits
}

func (m model) emptyTranscript(wrapAt int) []string {
	lines := []string{"", titleStyle.Render("Ready")}
	for _, line := range wrapText("Enter sends. Shift+Enter starts a new line.", wrapAt) {
		lines = append(lines, dimStyle.Render(line))
	}
	hint := "Ask a question.  /tools   /skills   /skill name   /system"
	if m.skill != "" {
		hint = "Skill " + m.skill + " is on.  " + hint
	}
	for _, line := range wrapText(hint, wrapAt) {
		lines = append(lines, dimStyle.Render(line))
	}
	return lines
}

func say(labelStyle lipgloss.Style, label, text string, width, wrapAt int, live bool) []string {
	at := wrapAt
	if live && at > 10 {
		at -= 2
	}
	lines := []string{labelStyle.Render(clip(label, width))}
	var body []string
	if live && strings.TrimSpace(text) == "" {
		body = []string{accent.Render("▍")}
	} else {
		body = wrapText(text, at)
		if live && len(body) > 0 {
			body[len(body)-1] += accent.Render(" ▍")
		}
	}
	for _, line := range body {
		lines = append(lines, gutter.Render("│ ")+line)
	}
	return lines
}

func (m model) page() int {
	n := m.height / 2
	if n < 1 {
		n = 1
	}
	return n
}

func (m model) verb(name, arg string) (model, string, error) {
	switch name {
	case "tools":
		return m, agent.ToolsText(), nil
	case "skills":
		return m, agent.SkillsText(), nil
	case "skill":
		if strings.TrimSpace(arg) == "" {
			m.skill = ""
			hodge.UseSkill("")
			return m, "Skill off.", nil
		}
		s, err := agent.Load(arg)
		if err != nil {
			return m, "", err
		}
		m.skill = s.Name
		hodge.UseSkill(s.Body)
		return m, "Skill " + s.Name + ".", nil
	case "system":
		text, err := hodge.UseSystem(arg)
		if err != nil {
			return m, "", err
		}
		return m, text, nil
	default:
		return m, "", nil
	}
}

func runTool(ctx context.Context, box *vm.Client, call *agent.Call) (label, out string, ok bool) {
	switch call.Tool {
	case "exec":
		if call.Cmd() == "" {
			return "sandbox", "Missing command.", true
		}
		text, _ := box.Exec(ctx, call.Cmd())
		return "sandbox", "$ " + call.Cmd() + "\n" + text, true
	case "search":
		text, err := search.New().Search(ctx, call.Query(), call.Source())
		if text == "" && err != nil {
			text = "The lookup failed."
		}
		return "search", text, true
	case "gather":
		text, err := search.New().Gather(ctx, call.URL())
		if text == "" && err != nil {
			text = "The lookup failed."
		}
		return "gather", text, true
	default:
		return "", "", false
	}
}

func wrapText(s string, width int) []string {
	if width < 8 {
		width = 8
	}
	var out []string
	for _, para := range strings.Split(s, "\n") {
		out = append(out, wrapLine(para, width)...)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func wrapLine(s string, width int) []string {
	r := []rune(s)
	if len(r) == 0 {
		return []string{""}
	}
	var out []string
	for len(r) > width {
		cut := width
		for i := cut; i > width/2; i-- {
			if r[i] == ' ' {
				cut = i
				break
			}
		}
		out = append(out, string(r[:cut]))
		r = r[cut:]
		for len(r) > 0 && r[0] == ' ' {
			r = r[1:]
		}
	}
	out = append(out, string(r))
	return out
}

func clip(s string, width int) string {
	if width < 4 || lipgloss.Width(s) <= width {
		return s
	}
	r := []rune(s)
	for len(r) > 0 && lipgloss.Width(string(r)) > width-1 {
		r = r[:len(r)-1]
	}
	return string(r) + "…"
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}
