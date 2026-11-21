package agent

import (
	"strings"
	"testing"
)

func TestParseExec(t *testing.T) {
	c := Parse("{\"type\":\"tool_call\",\"tool\":\"exec\",\"args\":{\"cmd\":\"ls -la\"}}")
	if c == nil || c.Tool != "exec" || c.Cmd() != "ls -la" {
		t.Fatalf("%#v", c)
	}
	if Slash("/run ls -la").Cmd() != "ls -la" {
		t.Fatal("slash")
	}
	if name, _, ok := Verb("/tools"); !ok || name != "tools" {
		t.Fatal("tools")
	}
	if name, arg, ok := Verb("/skill recipe"); !ok || name != "skill" || arg != "recipe" {
		t.Fatalf("skill %s %s", name, arg)
	}
	if _, _, ok := Verb("/skills"); !ok {
		t.Fatal("skills")
	}
	if name, arg, ok := Verb("/system be brief"); !ok || name != "system" || arg != "be brief" {
		t.Fatalf("system %s %q", name, arg)
	}
	if Parse("Sure, here is a recipe.") != nil {
		t.Fatal("prose")
	}
	s := Parse("{\"type\":\"tool_call\",\"tool\":\"search\",\"args\":{\"q\":\"lake champlain\"}}")
	if s == nil || s.Tool != "search" || s.Query() != "lake champlain" {
		t.Fatalf("search %#v", s)
	}
	if !strings.Contains(LookupWire("1. hit"), "Do not use JSON.") {
		t.Fatal("lookup wire")
	}
	page := "I'll read that page.\n{\"type\":\"tool_call\",\"tool\":\"gather\",\"args\":{\"url\":\"https://example.com/a\"}}"
	g := CallIn(page)
	if g == nil || g.Tool != "gather" || g.URL() != "https://example.com/a" {
		t.Fatalf("gather %#v", g)
	}
	if WithoutCall(page) != "I'll read that page." {
		t.Fatalf("lead %q", WithoutCall(page))
	}
	bare := CallIn("Gather this: {\"type\":\"tool_call\",\"tool\":\"gather\",\"args\":{}} https://example.com/b")
	if bare == nil || bare.URL() != "https://example.com/b" {
		t.Fatalf("url %#v", bare)
	}
}
