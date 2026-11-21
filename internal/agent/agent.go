// Package agent is the site's tool turn: a reply that is one tool call,
// then the result goes back into the chat.
package agent

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"sync"
)

const MaxRounds = 4

// Call is one tool call object.
type Call struct {
	Tool string
	Args map[string]any
}

// Cmd is the sandbox command, when the tool is exec.
func (c Call) Cmd() string {
	if c.Args == nil {
		return ""
	}
	for _, k := range []string{"cmd", "command", "q"} {
		if s, ok := c.Args[k].(string); ok {
			return strings.TrimSpace(s)
		}
	}
	return ""
}

// Slash is a direct command, the site's /run.
func Slash(text string) *Call {
	s := strings.TrimSpace(text)
	rest, ok := slashRest(s, "/run")
	if !ok {
		rest, ok = slashRest(s, "/exec")
	}
	if !ok || rest == "" {
		return nil
	}
	return &Call{Tool: "exec", Args: map[string]any{"cmd": rest}}
}

func slashRest(s, verb string) (string, bool) {
	if len(s) < len(verb) || !strings.EqualFold(s[:len(verb)], verb) {
		return "", false
	}
	if len(s) == len(verb) {
		return "", true
	}
	switch s[len(verb)] {
	case ' ', '\t', '\n':
		return strings.TrimSpace(s[len(verb):]), true
	default:
		return "", false
	}
}

// Alone is a reply that is one tool call and no other words.
func Alone(text string) *Call {
	s := strings.TrimSpace(text)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	return parseObject(strings.TrimSpace(s))
}

// CallIn is the tool call in a reply. The reply may also contain words.
// Alone stays the check for a reply that is nothing but the call.
func CallIn(text string) *Call {
	for _, obj := range objects(text) {
		if c := parseObject(obj); c != nil && knownTool(c.Tool) {
			c.complete(text)
			return c
		}
	}
	if c := Alone(text); c != nil && knownTool(c.Tool) {
		c.complete(text)
		return c
	}
	return nil
}

func knownTool(tool string) bool {
	switch strings.ToLower(strings.TrimSpace(tool)) {
	case "exec", "search", "gather":
		return true
	default:
		return false
	}
}

func (c *Call) complete(text string) {
	if c == nil || c.Args == nil {
		return
	}
	if strings.EqualFold(c.Tool, "gather") && c.URL() == "" {
		if u := firstURL(text); u != "" {
			c.Args["url"] = u
		}
	}
}

var urlIn = regexp.MustCompile(`https?://[^\s<>"'` + "`" + `)\]]+`)

func firstURL(text string) string {
	m := urlIn.FindString(text)
	return strings.TrimRight(m, ".,;:!?")
}

// WithoutCall is the reply with the tool call removed.
func WithoutCall(text string) string {
	c := CallIn(text)
	if c == nil {
		return strings.TrimSpace(text)
	}
	s := text
	if obj := objectOf(s); obj != "" {
		s = strings.Replace(s, obj, "", 1)
	}
	s = strings.ReplaceAll(s, "```json", "")
	s = strings.ReplaceAll(s, "```", "")
	return strings.TrimSpace(s)
}

func objectOf(text string) string {
	for _, obj := range objects(text) {
		if c := parseObject(obj); c != nil && knownTool(c.Tool) {
			return obj
		}
	}
	return ""
}

func objects(text string) []string {
	var out []string
	in := false
	esc := false
	for i := 0; i < len(text); i++ {
		c := text[i]
		if in {
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				in = false
			}
			continue
		}
		if c == '"' {
			in = true
			continue
		}
		if c != '{' {
			continue
		}
		obj := takeObject(text[i:])
		if obj == "" {
			break
		}
		out = append(out, obj)
		i += len(obj) - 1
	}
	return out
}

func takeObject(s string) string {
	if s == "" || s[0] != '{' {
		return ""
	}
	depth := 0
	in := false
	esc := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if in {
			if esc {
				esc = false
				continue
			}
			if c == '\\' {
				esc = true
				continue
			}
			if c == '"' {
				in = false
			}
			continue
		}
		if c == '"' {
			in = true
			continue
		}
		if c == '{' {
			depth++
			continue
		}
		if c == '}' {
			depth--
			if depth == 0 {
				return s[:i+1]
			}
		}
	}
	return ""
}

// Parse is the tool call in a finished reply, or nil.
func Parse(text string) *Call {
	s := strings.TrimSpace(text)
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(strings.TrimSpace(s), "```")
	s = strings.TrimSpace(s)
	if c := parseObject(s); c != nil {
		return c
	}
	start := strings.Index(text, "{")
	end := strings.LastIndex(text, "}")
	if start >= 0 && end > start {
		return parseObject(text[start : end+1])
	}
	return nil
}

func parseObject(s string) *Call {
	if !strings.HasPrefix(s, "{") || !strings.HasSuffix(s, "}") {
		return nil
	}
	var obj struct {
		Type string         `json:"type"`
		Tool string         `json:"tool"`
		Name string         `json:"name"`
		Args map[string]any `json:"args"`
	}
	if json.Unmarshal([]byte(s), &obj) != nil {
		return nil
	}
	tool := obj.Tool
	if tool == "" {
		tool = obj.Name
	}
	if tool == "" {
		return nil
	}
	if obj.Type != "" && obj.Type != "tool_call" {
		return nil
	}
	if obj.Args == nil {
		obj.Args = map[string]any{}
	}
	return &Call{Tool: tool, Args: obj.Args}
}

// Query is the search text.
func (c Call) Query() string {
	return c.arg("q", "query")
}

// URL is the page address for gather.
func (c Call) URL() string {
	return c.arg("url", "href", "address", "link")
}

// Source is a named lookup, such as wiki or duckduckgo.
func (c Call) Source() string {
	return c.arg("source")
}

func (c Call) arg(keys ...string) string {
	if c.Args == nil {
		return ""
	}
	for _, k := range keys {
		if s, ok := c.Args[k].(string); ok {
			return strings.TrimSpace(s)
		}
	}
	return ""
}

// Wire is the next user turn after a sandbox command, the same sentence the site sends.
func Wire(output string) string {
	return "Results:\n" + output + "\nAnswer the person in plain words, or reply with one more tool call if the task needs it."
}

// LookupWire is the next user turn after search or gather. The site forbids another JSON call there.
func LookupWire(output string) string {
	return "Results:\n" + output + "\nAnswer the person in plain words. Do not use JSON."
}

var (
	armMu sync.Mutex
	arm   context.CancelFunc
)

// Arm remembers the sandbox call so Ctrl+C can cancel it.
func Arm(cancel context.CancelFunc) {
	armMu.Lock()
	arm = cancel
	armMu.Unlock()
}

// Cancel stops the sandbox call that is in flight.
func Cancel() {
	armMu.Lock()
	c := arm
	armMu.Unlock()
	if c != nil {
		c()
	}
}
