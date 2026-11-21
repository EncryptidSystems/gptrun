// Command gptrun runs the Hodge-Kuramoto runtime. A packed linear is
// hodgelinear.wgsl on Dawn, limbs included. A dense bf16 linear is dense.wgsl.
// LUT kernels are the WGSL files Dawn loads as written. The config stamp
// model_type is not the runtime name.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"gptrun/internal/agent"
	"gptrun/internal/api"
	"gptrun/internal/catalog"
	"gptrun/internal/generate"
	"gptrun/internal/hodge"
	"gptrun/internal/mark"
	"gptrun/internal/rhfl"
	"gptrun/internal/search"
	"gptrun/internal/spell"
	"gptrun/internal/tui"
	"gptrun/internal/version"
	"gptrun/internal/vm"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func gdesk(args []string) error {
	fs := flag.NewFlagSet("gdesk", flag.ContinueOnError)
	vocab := fs.String("vocab", `C:\Users\grego\models\prod\2b4t\vocab.float.safetensors`, "vocab.float.safetensors")
	tok := fs.String("tokenizer", `C:\Users\grego\models\prod\2b4t\tokenizer.json`, "tokenizer.json")
	corpus := fs.String("corpus", `C:\Users\grego\watermarks`, "folder of .txt texts")
	addr := fs.String("addr", "127.0.0.1:58165", "listen address")
	if err := fs.Parse(args); err != nil {
		return err
	}
	d, err := mark.OpenGDesk(*vocab, *tok, *corpus)
	if err != nil {
		return err
	}
	return d.Serve(*addr)
}

func run(args []string) error {
	if len(args) == 0 {
		fmt.Print(usage)
		return nil
	}
	if args[0] == "run" || args[0] == "serve" {
		if msg, stale := version.Stale(); stale {
			fmt.Fprintln(os.Stderr, "warning: "+msg)
		}
	}
	switch args[0] {
	case "help", "-h", "--help":
		fmt.Print(usage)
		return nil
	case "eval":
		return evalCmd(args[1:])
	case "bench":
		return benchCmd(args[1:])
	case "version", "-v", "--version":
		fmt.Println(version.String())
		return nil
	case "list", "ls":
		return list()
	case "ps":
		return ps()
	case "show":
		if len(args) < 2 {
			return fmt.Errorf("gptrun show <model>")
		}
		return show(args[1])
	case "gdesk":
		return gdesk(args[1:])
	case "hes":
		return hesCmd(args[1:])
	case "serve":
		addr := "127.0.0.1:11434"
		for i := 1; i < len(args); i++ {
			if args[i] == "-addr" && i+1 < len(args) {
				addr = args[i+1]
				i++
			}
		}
		hodge.SetLog(func(s string) { fmt.Fprintln(os.Stderr, s) })
		fmt.Println(version.String(), "serving on", addr)
		return http.ListenAndServe(addr, api.Handler())
	case "run":
		name, prompt, config, opt, err := parseRun(args[1:])
		if err != nil {
			return err
		}
		if prompt == "" && isTTY() {
			if !opt.TempSet && !opt.TopSet {
				opt.ConfigSample = true
			}
			if config != "" {
				return tui.RunFile(config, opt)
			}
			return tui.Run(name, opt)
		}
		if prompt == "" {
			b, err := io.ReadAll(os.Stdin)
			if err != nil || len(strings.TrimSpace(string(b))) == 0 {
				return fmt.Errorf("gptrun run %s: prompt missing", name)
			}
			prompt = string(b)
		}
		if text, ok := verbText(prompt); ok {
			fmt.Println(text)
			return nil
		}
		hodge.ClearInterrupt()
		hodge.SetLog(func(s string) { fmt.Fprintln(os.Stderr, s) })
		text, _, err := answer(name, config, prompt, opt, true, func(s string) { fmt.Fprintln(os.Stderr, s) })
		if err != nil {
			return err
		}
		printText(text)
		reportMark(name, config, text)
		return nil
	default:
		return fmt.Errorf("unknown %q\n%s", args[0], usage)
	}
}

// answer is the reply `gptrun run` prints for prompt and the route it
// took. The prompt goes straight to chat; no think pass runs first. A chat
// reply that turns out to need a tool gets one when tools is set (the reply
// otherwise stands, and the route still says command), and one that needs a
// lookup gets one. status hears progress lines.
func answer(name, config, prompt string, opt generate.Opt, tools bool, status func(string)) (string, agent.Route, error) {
	target := name
	if config != "" {
		target = config
	}
	if fixed, ok := spell.Adapt(tokenizerOf(name, config), prompt); ok {
		status("spell  " + fixed)
		prompt = fixed
	}
	turns := []generate.Turn{{Role: "user", Content: prompt}}
	var rep generate.Reply
	var err error
	if config != "" {
		rep, err = generate.ChatFile(config, turns, opt, nil)
	} else {
		rep, err = generate.Chat(name, turns, opt, nil)
	}
	if err != nil {
		return "", agent.Talk, err
	}
	text := rep.Text
	switch kind, _ := rhfl.Of(prompt, text); kind {
	case rhfl.Tool:
		if !tools {
			return text, agent.Command, nil
		}
		text, err = toolTurns(target, config, prompt, text, opt)
		if err != nil {
			return "", agent.Command, err
		}
		return text, agent.Command, nil
	case rhfl.Wiki:
		user, sources, gerr := search.New().Ground(context.Background(), prompt, status, nil)
		if gerr != nil {
			return "", agent.Fact, gerr
		}
		shown, serr := generate.Shown(target, prompt, agent.Fact, user, sources, opt)
		if serr != nil {
			return "", agent.Fact, serr
		}
		if strings.TrimSpace(shown) != "" {
			text = shown
		}
		return text, agent.Fact, nil
	}
	return text, agent.Talk, nil
}

func printText(text string) {
	if text == "" {
		fmt.Println(text)
	} else if !strings.HasSuffix(text, "\n") {
		fmt.Println(text)
	} else {
		fmt.Print(text)
	}
	if strings.TrimSpace(text) == "" {
		return
	}
	if line := hodge.SpeedLine(); line != "" {
		fmt.Println(line)
	}
}

func tokenizerOf(name, config string) string {
	if config != "" {
		if m, err := catalog.Load(config); err == nil {
			return m.Tokenizer
		}
	}
	if m, ok, err := catalog.Get(name); err == nil && ok {
		return m.Tokenizer
	}
	return ""
}

func toolTurns(target, config, question, reply string, opt generate.Opt) (string, error) {
	ctx := context.Background()
	box := vm.New()
	web := search.New()
	turns := []generate.Turn{
		{Role: "user", Content: question},
		{Role: "assistant", Content: reply},
	}
	text := reply
	for round := 0; round < agent.MaxRounds; round++ {
		call := agent.CallIn(text)
		if call == nil {
			break
		}
		fmt.Fprintf(os.Stderr, "%s…\n", call.Tool)
		var out string
		switch call.Tool {
		case "exec":
			out, _ = box.Exec(ctx, call.Cmd())
			out = "$ " + call.Cmd() + "\n" + out
		case "search":
			out, _ = web.Search(ctx, call.Query(), call.Source())
		case "gather":
			out, _ = web.Gather(ctx, call.URL())
		default:
			return text, nil
		}
		fmt.Fprintf(os.Stderr, "%s\n", out)
		next := agent.Wire(out)
		if call.Tool != "exec" {
			next = agent.LookupWire(out)
		}
		turns = append(turns, generate.Turn{Role: "user", Content: next})
		var rep generate.Reply
		var err error
		if config != "" {
			rep, err = generate.ChatFile(config, turns, opt, nil)
		} else {
			rep, err = generate.Chat(target, turns, opt, nil)
		}
		if err != nil {
			return "", err
		}
		text = rep.Text
		turns = append(turns, generate.Turn{Role: "assistant", Content: text})
	}
	if agent.CallIn(text) == nil {
		return agent.WithoutCall(text), nil
	}
	turns = append(turns, generate.Turn{Role: "user", Content: "Answer the person in plain words. Do not use JSON."})
	var rep generate.Reply
	var err error
	if config != "" {
		rep, err = generate.ChatFile(config, turns, opt, nil)
	} else {
		rep, err = generate.Chat(target, turns, opt, nil)
	}
	if err != nil {
		return "", err
	}
	return rep.Text, nil
}

func verbText(prompt string) (string, bool) {
	name, arg, ok := agent.Verb(prompt)
	if !ok {
		return "", false
	}
	switch name {
	case "tools":
		return agent.ToolsText(), true
	case "skills":
		return agent.SkillsText(), true
	case "skill":
		if strings.TrimSpace(arg) == "" {
			hodge.UseSkill("")
			return "Skill off.", true
		}
		s, err := agent.Load(arg)
		if err != nil {
			return err.Error(), true
		}
		hodge.UseSkill(s.Body)
		return "Skill " + s.Name + ".\n\n" + s.Body, true
	case "system":
		text, err := hodge.UseSystem(arg)
		if err != nil {
			return err.Error(), true
		}
		return text, true
	default:
		return "", false
	}
}

func parseRun(args []string) (name, prompt, config string, opt generate.Opt, err error) {
	opt = generate.Opt{ConfigSample: true}
	i := 0
	for i < len(args) {
		a := args[i]
		if a == "--" {
			i++
			break
		}
		if !strings.HasPrefix(a, "-") || a == "-" {
			break
		}
		switch a {
		case "-n":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -n needs a number")
			}
			n, e := strconv.Atoi(args[i+1])
			if e != nil || n < 1 {
				return "", "", "", opt, fmt.Errorf("gptrun run: -n %q", args[i+1])
			}
			opt.N = n
			i += 2
		case "-config", "--config":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: --config needs a file")
			}
			config = args[i+1]
			i += 2
		case "-temp":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -temp needs a number")
			}
			t, e := strconv.ParseFloat(args[i+1], 64)
			if e != nil || t < 0 {
				return "", "", "", opt, fmt.Errorf("gptrun run: -temp %q", args[i+1])
			}
			opt.Temp = t
			opt.TempSet = true
			i += 2
		case "-top-p":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -top-p needs a number")
			}
			p, e := strconv.ParseFloat(args[i+1], 64)
			if e != nil || p < 0 {
				return "", "", "", opt, fmt.Errorf("gptrun run: -top-p %q", args[i+1])
			}
			opt.TopP = p
			opt.TopSet = true
			i += 2
		case "-system", "--system":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -system needs the prompt or a file")
			}
			if _, e := hodge.UseSystem(args[i+1]); e != nil {
				return "", "", "", opt, e
			}
			i += 2
		case "-chat", "--chat":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -chat needs a format (role, llama3, gemma, qwen, harmony, plain)")
			}
			if e := hodge.UseChat(args[i+1]); e != nil {
				return "", "", "", opt, fmt.Errorf("gptrun run: %w", e)
			}
			i += 2
		case "-memory", "--memory":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -memory needs a file")
			}
			if _, e := os.Stat(args[i+1]); e != nil {
				return "", "", "", opt, fmt.Errorf("gptrun run: -memory: %w", e)
			}
			opt.Memory = args[i+1]
			i += 2
		case "-image", "--image":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -image needs a file")
			}
			if _, e := os.Stat(args[i+1]); e != nil {
				return "", "", "", opt, fmt.Errorf("gptrun run: -image: %w", e)
			}
			opt.Image = args[i+1]
			i += 2
		case "-hes", "--hes":
			if i+1 >= len(args) {
				return "", "", "", opt, fmt.Errorf("gptrun run: -hes needs a state file (made on first use)")
			}
			hodge.UseHESFile(args[i+1])
			i += 2
		default:
			return "", "", "", opt, fmt.Errorf("unknown flag %q\n%s", a, usage)
		}
	}
	if config != "" {
		prompt = strings.Join(args[i:], " ")
		return "", prompt, config, opt, nil
	}
	if i >= len(args) {
		return "", "", "", opt, fmt.Errorf("gptrun run --config FILE [prompt]")
	}
	name = args[i]
	prompt = strings.Join(args[i+1:], " ")
	return name, prompt, "", opt, nil
}

func list() error {
	all, err := catalog.List()
	if err != nil {
		return err
	}
	for _, m := range all {
		extra := m.Detail
		if m.Family == catalog.FamilyHodge && m.Layers > 0 {
			extra = fmt.Sprintf("%s  %s  %d layers  hidden %d", m.Detail, m.Kind, m.Layers, m.Hidden)
		}
		fmt.Printf("%-28s %-8s %s\n", m.Name, m.Family, extra)
	}
	return nil
}

func show(name string) error {
	m, ok, err := catalog.Get(name)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("no model %q", name)
	}
	fmt.Printf("name     %s\nfamily   %s\nkind     %s\n", m.Name, m.Family, m.Kind)
	if m.Detail != "" {
		fmt.Printf("detail   %s\n", m.Detail)
	}
	if m.Layers > 0 {
		fmt.Printf("layers   %d\nhidden   %d\nvocab    %d\n", m.Layers, m.Hidden, m.Vocab)
	}
	if m.Bytes > 0 {
		fmt.Printf("bytes    %d\n", m.Bytes)
	}
	if m.Path != "" {
		fmt.Printf("path     %s\n", m.Path)
	}
	return nil
}

func reportMark(name, config, text string) {
	var tok, cfg string
	if config != "" {
		if m, err := catalog.Load(config); err == nil {
			tok, cfg = m.Tokenizer, m.TokenizerConfig
		}
	} else if m, err := catalog.Load(name); err == nil {
		tok, cfg = m.Tokenizer, m.TokenizerConfig
	} else if m, ok, err := catalog.Get(name); err == nil && ok {
		tok, cfg = m.Tokenizer, m.TokenizerConfig
	}
	c, err := mark.Cached(tok, cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "watermark:", err)
		return
	}
	if c == nil || strings.TrimSpace(text) == "" {
		return
	}
	s := c.Score(text)
	if s.Words == 0 {
		return
	}
	kind := "clear"
	if s.Marked {
		kind = "marked"
	} else if s.Words < 24 {
		kind = "short"
	}
	fmt.Fprintf(os.Stderr, "watermark  %s  words %d  alignment %.3f  null %.3f  cell %.4f  cell_null %.4f\n", kind, s.Words, s.Alignment, s.NullMax, s.CellShare, s.CellNull)
}

func ps() error {
	got := generate.Loaded()
	if len(got) == 0 {
		fmt.Println("no models loaded")
		return nil
	}
	for name, at := range got {
		fmt.Printf("%s  %s\n", name, at.Format("15:04:05"))
	}
	return nil
}

func isTTY() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

const usage = `gptrun runs the Hodge-Kuramoto runtime. A packed linear is hodgelinear.wgsl
on Dawn, limbs included. A dense bf16 linear is dense.wgsl. LUT kernels are the
WGSL files Dawn loads as written. The config stamp model_type is not the runtime name.

  gptrun list
  gptrun run [-n N] [-temp T] [-top-p P] [-system TEXT] [-chat FORMAT] [-image FILE] [-memory FILE] [-hes STATE] --config FILE [text]
  gptrun show <model>
  gptrun ps
  gptrun serve [-addr 127.0.0.1:11434]
  gptrun gdesk [-addr 127.0.0.1:58165]
  gptrun hes [-state FILE] [-dims 64x64] [-writes N] [-lambda L] [-window W] [-beta B] [-gpu=false]
  gptrun eval [-set FILE] [-min F] (--config FILE | <model>)
  gptrun bench --config FILE [-task arc-challenge,arc-easy,openbookqa] [-limit N] [-out FILE]
  gptrun version

Omit -n to run until the model stops. -n N caps the answer.
Memory is the holonomy–eikonal state: a prompt, a history or a reply longer than
the window goes through the field instead of failing. -memory FILE writes its
passages ("---" between them) into a field and reads back what fits. -hes STATE
keeps the chat field in STATE across runs. HES_DIMS sets the torus (64x64).
Omit -temp to use the checkpoint sampler. -temp 0 is greedy.
The HTTP routes match Ollama: /api/tags /api/chat /api/generate /api/show /api/ps.
When model.cochain.safetensors sits beside the tokenizer, each reply is scored
against the vocabulary as it is produced. The HTTP object carries watermark.
gptrun run prints the same line on stderr.
`
