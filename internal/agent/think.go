package agent

import "strings"

// Route is what a question is, from a think pass. It is not read off the
// person's wording.
type Route string

const (
	// Fact is a checkable fact a reference page can settle.
	Fact Route = "fact"
	// Recipe is a dish. The model writes the ingredients and the method.
	Recipe Route = "recipe"
	// Command is a command or a file listing in the guest.
	Command Route = "command"
	// Talk is everything the model should answer itself.
	Talk Route = "talk"
)

// RouteOnly is a reply that is nothing but a route word, such as "Recipe."
// or "Fact". That word names the question. It is not the answer.
func RouteOnly(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" || strings.Contains(s, "\n") {
		return false
	}
	_, ok := labelLine(s)
	return ok
}

// RecipeSystem is the answer pass for a dish. It asks for the method.
const RecipeSystem = "Write the ingredients and the method. Use plain sentences. Start with the ingredients. A label is not the answer."

// RecipeRetry is the second pass when the first reply was only the word recipe.
const RecipeRetry = "List the ingredients, then the steps. Start with the first ingredient."

// CommandSystem is the tool pass for a guest command. The reply is one call.
const CommandSystem = "Do not answer in words. Reply with one tool call and nothing else. " +
	"Use {\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"the command\"}}. " +
	"The command runs in the guest, not on this machine."

// FactSystem is the answer pass after a lookup. The device system says to
// keep the answer short and to stop in one sentence when unsure. That is
// how the reply never answers the question that was asked.
const FactSystem = "Answer the question the person asked. Use the notes. A label is not the answer."

// Named reports that the think pass put a route word on its own line.
func Named(thought string) bool {
	for _, line := range strings.Split(thought, "\n") {
		if _, ok := labelLine(line); ok {
			return true
		}
	}
	return false
}

// FactWhenUnnamed is a checkable question whose think pass never named a
// route. A line that names talk, recipe, command, or fact stands. A recipe
// stays with the model: a page about the dish is not the method. A model
// that answers the question instead of naming a route still gets the lookup.
func FactWhenUnnamed(question, thought string) bool {
	if Named(thought) {
		return false
	}
	q := strings.TrimSpace(question)
	if recipe.MatchString(q) {
		return false
	}
	return NeedsFacts(q)
}

// RouteOf reads the route word from a think pass. A reflection that never
// names one of the four words is talk.
func RouteOf(thought string) Route {
	for _, line := range strings.Split(thought, "\n") {
		if route, ok := labelLine(line); ok {
			return route
		}
	}
	return Talk
}

// RouteClosed reports that the text ends on a finished route line.
// The newline after the word is what closes it. A route word with no
// newline is still being written.
func RouteClosed(s string) bool {
	if !strings.HasSuffix(s, "\n") {
		return false
	}
	body := strings.TrimRight(s, "\r\n")
	if body == "" {
		return false
	}
	lines := strings.Split(body, "\n")
	_, ok := labelLine(lines[len(lines)-1])
	return ok
}

func labelLine(s string) (Route, bool) {
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) == 2 && strings.EqualFold(strings.Trim(fields[0], ":"), "route") {
		return known(fields[1])
	}
	if len(fields) != 1 {
		return "", false
	}
	return known(fields[0])
}

// Reasoning is the thought without a trailing route word.
func Reasoning(thought string) string {
	lines := strings.Split(strings.TrimSpace(thought), "\n")
	for len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) == "" {
		lines = lines[:len(lines)-1]
	}
	if len(lines) > 0 {
		if _, ok := labelLine(lines[len(lines)-1]); ok {
			lines = lines[:len(lines)-1]
		}
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// AnswerAsk is the user turn for the answer pass. The thought is context.
// It is not part of the reply.
func AnswerAsk(question, thought string) string {
	q := strings.TrimSpace(question)
	t := Reasoning(thought)
	if t == "" {
		return q
	}
	return "Question: " + q + "\n\nThought, already written:\n" + t +
		"\n\nAnswer the question. Do not repeat the thought."
}

// AnswerOnly removes a leading copy of the thought from the reply.
func AnswerOnly(thought, reply string) string {
	reply = strings.TrimSpace(reply)
	reply = strings.TrimSpace(strings.TrimPrefix(reply, "Answer:"))
	thought = Reasoning(thought)
	if reply == "" || thought == "" {
		return reply
	}
	if foldHasPrefix(squash(reply), squash(thought)) {
		return strings.TrimSpace(dropFoldPrefix(reply, thought))
	}
	body := squash(thought)
	lines := strings.Split(reply, "\n")
	i := 0
	for i < len(lines) {
		line := strings.TrimSpace(lines[i])
		if line == "" || strings.EqualFold(line, "Answer:") {
			i++
			continue
		}
		if len(line) >= 24 && strings.Contains(body, squash(line)) {
			i++
			continue
		}
		break
	}
	return strings.TrimSpace(strings.Join(lines[i:], "\n"))
}

func squash(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

func foldHasPrefix(text, prefix string) bool {
	return prefix != "" && strings.HasPrefix(text, prefix)
}

// dropFoldPrefix drops a leading copy of prefix, ignoring case and the
// space between words. The remainder is the original reply.
func dropFoldPrefix(reply, prefix string) string {
	ri, pi := 0, 0
	rb, pb := []rune(reply), []rune(prefix)
	for pi < len(pb) && ri < len(rb) {
		pr, rr := pb[pi], rb[ri]
		if isSpace(pr) {
			pi++
			for ri < len(rb) && isSpace(rb[ri]) {
				ri++
			}
			continue
		}
		if isSpace(rr) {
			ri++
			continue
		}
		if fold(pr) != fold(rr) {
			return strings.TrimSpace(reply)
		}
		pi++
		ri++
	}
	if pi < len(pb) {
		return strings.TrimSpace(reply)
	}
	return string(rb[ri:])
}

func isSpace(r rune) bool {
	return r == ' ' || r == '\n' || r == '\t' || r == '\r'
}

func fold(r rune) rune {
	if r >= 'A' && r <= 'Z' {
		return r + ('a' - 'A')
	}
	return r
}

func known(s string) (Route, bool) {
	switch strings.ToLower(strings.Trim(s, "`*_\"'.:!")) {
	case "fact":
		return Fact, true
	case "recipe":
		return Recipe, true
	case "command":
		return Command, true
	case "talk":
		return Talk, true
	default:
		return "", false
	}
}
