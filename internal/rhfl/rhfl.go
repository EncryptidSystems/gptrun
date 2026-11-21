// Package rhfl routes a finished reply for correction.
//
// The model speaks first. Wikipedia and the guest tools stay in the runtime,
// and they run only when the reply is a residual: a tool call and no other
// words, a bare fact label, an empty checkable answer, or a short decline of
// a live or checkable fact. A recipe is not sent to a page. A conversational
// reply stands, including one that does not repeat the question's words.
package rhfl

import (
	"regexp"
	"strings"

	"gptrun/internal/agent"
	"gptrun/internal/search"
)

// Kind is the correction, if any.
type Kind string

const (
	// Stand is a reply the person should read as said.
	Stand Kind = ""
	// Wiki is a short lookup that replaces the reply.
	Wiki Kind = "wiki"
	// Tool is one tool call and no other words.
	Tool Kind = "tool"
)

// Of reads a finished reply. call is set only for Tool.
func Of(question, reply string) (Kind, *agent.Call) {
	if call := agent.CallIn(reply); call != nil {
		return Tool, call
	}
	if wiki(question, reply) {
		return Wiki, nil
	}
	return Stand, nil
}

func wiki(question, reply string) bool {
	if !checkable(question) {
		return false
	}
	reply = strings.TrimSpace(reply)
	if reply == "" {
		return true
	}
	if agent.RouteOnly(reply) && agent.RouteOf(reply) == agent.Fact {
		return true
	}
	if !declined(reply) {
		return false
	}
	if agent.Live(question) {
		return true
	}
	return !search.CarriesName(question, reply)
}

// checkable is a fact a short lookup can settle. A dish stays with the model.
func checkable(question string) bool {
	if agent.Dish(question) {
		return false
	}
	return agent.Live(question) || agent.NeedsFacts(question)
}

var decline = regexp.MustCompile(`(?i)^(?:i(?:'m| am) not sure|i do not know|i don't know|i cannot say|i can't say|i(?:'m| am) unsure|not sure)\b`)

func declined(reply string) bool {
	s := strings.TrimSpace(reply)
	if s == "" || len(strings.Fields(s)) > 16 {
		return false
	}
	return decline.MatchString(s)
}
