// Package generate is the one reply path for the TUI and the HTTP API.
// Hodge-Kuramoto checkpoints go through hodgelinear on Dawn. LUT kernels
// and the Dawn clock stay on the one-shot dawn.exe script.
package generate

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"gptrun/internal/agent"
	"gptrun/internal/catalog"
	"gptrun/internal/dawn"
	"gptrun/internal/hodge"
	"gptrun/internal/rag"
	"gptrun/internal/search"
)

// Reply is one assistant turn.
type Reply struct {
	Model   string
	Text    string
	Elapsed time.Duration
}

// Turn is one chat message.
type Turn = hodge.Turn

// Opt is the sampler passed to the Hodge-Kuramoto runtime.
type Opt = hodge.Opt

var (
	mu     sync.Mutex
	loaded = map[string]time.Time{}
)

// Loaded is the set of models that have run in this process.
func Loaded() map[string]time.Time {
	mu.Lock()
	defer mu.Unlock()
	out := make(map[string]time.Time, len(loaded)+1)
	for k, v := range loaded {
		out[k] = v
	}
	if name, ok := hodge.Loaded(); ok {
		if _, seen := out[name]; !seen {
			out[name] = time.Now()
		}
	}
	return out
}

func withLibrary(turns []Turn) []Turn {
	out := append([]Turn(nil), turns...)
	for i := len(out) - 1; i >= 0; i-- {
		if strings.EqualFold(out[i].Role, "user") {
			out[i].Content = rag.Augment(out[i].Content)
			break
		}
	}
	return out
}

func mark(name string) {
	mu.Lock()
	loaded[name] = time.Now()
	mu.Unlock()
}

// Run answers prompt as one chat turn with the checkpoint's sampler and 32 tokens.
func Run(name, prompt string) (Reply, error) {
	return Chat(name, []Turn{{Role: "user", Content: prompt}}, Opt{ConfigSample: true}, nil)
}

// ChatFile is a stateless dialog for the checkpoint named by a config file.
func ChatFile(path string, turns []Turn, opt Opt, emit func(string)) (Reply, error) {
	start := time.Now()
	text, err := hodge.Transcript(path, withLibrary(turns), opt, emit)
	if err != nil {
		return Reply{Model: path, Elapsed: time.Since(start)}, err
	}
	mark(path)
	return Reply{Model: path, Text: text, Elapsed: time.Since(start)}, nil
}

// Chat is a stateless dialog. emit receives each decoded piece as it is produced.
func Chat(name string, turns []Turn, opt Opt, emit func(string)) (Reply, error) {
	m, err := lookup(name)
	if err != nil {
		return Reply{}, err
	}
	if m.Family != catalog.FamilyHodge {
		return once(m, rag.Augment(lastUser(turns)), emit)
	}
	start := time.Now()
	text, err := hodge.Transcript(name, withLibrary(turns), opt, emit)
	if err != nil {
		return Reply{Model: m.Name, Elapsed: time.Since(start)}, err
	}
	mark(m.Name)
	return Reply{Model: m.Name, Text: text, Elapsed: time.Since(start)}, nil
}

// Complete is a raw completion. The chat template is not applied.
func Complete(name, prompt string, opt Opt, emit func(string)) (Reply, error) {
	m, err := lookup(name)
	if err != nil {
		return Reply{}, err
	}
	if m.Family != catalog.FamilyHodge {
		return once(m, rag.Augment(prompt), emit)
	}
	start := time.Now()
	text, err := hodge.Raw(name, rag.Augment(prompt), opt, emit)
	if err != nil {
		return Reply{Model: m.Name, Elapsed: time.Since(start)}, err
	}
	mark(m.Name)
	return Reply{Model: m.Name, Text: text, Elapsed: time.Since(start)}, nil
}

// ExecCall asks for one guest command and does not write the TUI chat cache.
func ExecCall(name, question string, opt Opt) (string, error) {
	if strings.TrimSpace(question) == "" {
		return "", nil
	}
	return hodge.Transcript(name, []Turn{
		{Role: "system", Content: agent.CommandSystem},
		{Role: "user", Content: question},
	}, opt, nil)
}

// Shown is the reply a person should read. A route word is not a reply.
// grounded and sources come from a lookup. They are used only for a fact.
func Shown(name, question string, route agent.Route, grounded, sources string, opt Opt) (string, error) {
	var text string
	var err error
	switch route {
	case agent.Recipe:
		text, err = recipeText(name, question, opt)
	case agent.Fact:
		text, err = factText(name, question, grounded, opt)
	default:
		return "", fmt.Errorf("route %s", route)
	}
	if err != nil {
		return "", err
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return "", nil
	}
	if s := strings.TrimSpace(sources); route == agent.Fact && s != "" {
		text = text + "\n\n" + s
	}
	if err := hodge.Absorb(name, question, text); err != nil && err != hodge.ErrInterrupted {
		return text, nil
	}
	return text, nil
}

func recipeText(name, question string, opt Opt) (string, error) {
	text, err := hodge.Transcript(name, []Turn{
		{Role: "system", Content: agent.RecipeSystem},
		{Role: "user", Content: question},
	}, opt, nil)
	if err != nil || !agent.RouteOnly(text) {
		return text, err
	}
	text, err = hodge.Transcript(name, []Turn{
		{Role: "system", Content: agent.RecipeRetry},
		{Role: "user", Content: question},
	}, opt, nil)
	if err != nil || agent.RouteOnly(text) {
		return "", err
	}
	return text, nil
}

// aboutSystem keeps the preground on the subject. The device system would
// turn "Tell me about" into a tool call, and that call is not the notes.
const aboutSystem = "Tell the person about the subject in plain words. Do not call a tool. Do not use JSON."

// AboutThenAsk pregrounds a fact. The model writes "Tell me about" the
// subject first. That text is the notes. It then answers the question
// from those notes.
func AboutThenAsk(name, question string, opt Opt) (string, error) {
	q := strings.TrimSpace(question)
	about := agent.AboutPrompt(q)
	if about == "" {
		return "", nil
	}
	overview, err := hodge.Transcript(name, []Turn{
		{Role: "system", Content: aboutSystem},
		{Role: "user", Content: about},
	}, opt, nil)
	if err != nil {
		return "", err
	}
	overview = strings.TrimSpace(overview)
	if overview == "" {
		return "", nil
	}
	text, err := factText(name, q, notesAsk(q, overview), opt)
	if err != nil || strings.TrimSpace(text) == "" {
		return text, err
	}
	if err := hodge.Absorb(name, q, text); err != nil && err != hodge.ErrInterrupted {
		return text, nil
	}
	return text, nil
}

func notesAsk(question, notes string) string {
	return "The lookup is already done. Answer the question the person asked, in plain words. " +
		"Use the notes. Give the answer that question asks for. Do not answer with a label, do not use a heading, and do not use JSON.\nNotes:\n" +
		strings.TrimSpace(notes) + "\nQuestion: " + strings.TrimSpace(question)
}

func factText(name, question, grounded string, opt Opt) (string, error) {
	ask := strings.TrimSpace(grounded)
	if ask == "" {
		ask = question
	}
	text, err := hodge.Transcript(name, []Turn{
		{Role: "system", Content: agent.FactSystem},
		{Role: "user", Content: ask},
	}, Greedy(opt), nil)
	if err != nil {
		return "", err
	}
	return search.Settle(question, ask, text), nil
}

// Greedy is opt with sampling off. An answer from notes takes the model's
// best word each time; a draw at the config temperature adds names the notes
// never had.
func Greedy(opt Opt) Opt {
	opt.Temp, opt.TempSet = 0, true
	return opt
}

// Continue keeps the TUI's KV cache for this model.
func Continue(name, user string, opt Opt, emit func(string)) (Reply, error) {
	m, err := lookup(name)
	if err != nil {
		return Reply{}, err
	}
	if m.Family != catalog.FamilyHodge {
		return once(m, user, emit)
	}
	start := time.Now()
	text, err := hodge.Continue(name, user, opt, emit)
	if err != nil {
		return Reply{Model: m.Name, Elapsed: time.Since(start)}, err
	}
	mark(m.Name)
	return Reply{Model: m.Name, Text: text, Elapsed: time.Since(start)}, nil
}

func lookup(name string) (catalog.Model, error) {
	m, ok, err := catalog.Get(name)
	if err != nil {
		return catalog.Model{}, err
	}
	if !ok {
		return catalog.Model{}, fmt.Errorf("no model %q (gptrun list)", name)
	}
	return m, nil
}

func lastUser(turns []Turn) string {
	for i := len(turns) - 1; i >= 0; i-- {
		if strings.EqualFold(turns[i].Role, "user") {
			return turns[i].Content
		}
	}
	return ""
}

func once(m catalog.Model, prompt string, emit func(string)) (Reply, error) {
	var rep Reply
	var err error
	switch m.Family {
	case catalog.FamilyClock:
		rep, err = clock(m)
	default:
		rep, err = prog(m, prompt)
	}
	if err == nil && emit != nil && rep.Text != "" {
		emit(rep.Text)
	}
	return rep, err
}

func clock(m catalog.Model) (Reply, error) {
	res, err := dawn.Run("clock")
	if err != nil {
		return Reply{Model: m.Name}, err
	}
	mark(m.Name)
	return Reply{Model: m.Name, Text: strings.TrimSpace(res.Stdout), Elapsed: res.Elapsed}, nil
}

func prog(m catalog.Model, _ string) (Reply, error) {
	res, err := dawn.Run("prog " + m.Path)
	if err != nil {
		return Reply{Model: m.Name}, err
	}
	mark(m.Name)
	text := strings.TrimSpace(res.Stdout)
	if text == "" {
		text = m.Path
	}
	return Reply{Model: m.Name, Text: text, Elapsed: res.Elapsed}, nil
}
