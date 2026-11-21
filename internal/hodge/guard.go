package hodge

import "strings"

// guardrail is a trained refusal. Once that reply sits in the chat cache,
// later turns repeat it. The words stay on screen. The cache does not keep them.
func guardrail(text string) bool {
	s := strings.ToLower(strings.TrimSpace(text))
	s = strings.Trim(s, "\"'`“”")
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	for _, p := range []string{
		"i'm sorry, but i can't assist",
		"i'm sorry, but i cannot assist",
		"i am sorry, but i can't assist",
		"i am sorry, but i cannot assist",
		"sorry, but i can't assist",
		"sorry, but i cannot assist",
		"i can't assist with that",
		"i cannot assist with that",
	} {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// dropGuardrails removes a refusal and the user turn it answered.
// A later question is kept, so the model is not asked to continue the refusal.
func dropGuardrails(turns []Turn) []Turn {
	out := make([]Turn, 0, len(turns))
	for i := 0; i < len(turns); i++ {
		role := strings.ToLower(turns[i].Role)
		if role == "user" && i+1 < len(turns) &&
			strings.EqualFold(turns[i+1].Role, "assistant") &&
			guardrail(turns[i+1].Content) {
			i++
			continue
		}
		if role == "assistant" && guardrail(turns[i].Content) {
			continue
		}
		out = append(out, turns[i])
	}
	return out
}

// rewind drops tokens written at and after at.
// Attention reads positions 0..pos, so the next turn does not see them.
func (d *dialog) rewind(at int) {
	if d == nil || at < 0 || at > d.pos {
		return
	}
	d.pos = at
	if at <= len(d.ids) {
		d.ids = d.ids[:at]
	}
	d.lg = nil
}
