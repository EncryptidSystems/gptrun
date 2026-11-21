package agent

import "testing"

func TestAnswerOnlyDropsTheThought(t *testing.T) {
	thought := "The district was placed on the Potomac by the Residence Act.\ntalk"
	reply := "The district was placed on the Potomac by the Residence Act.\n\nCongress passed it on July 16, 1790."
	got := AnswerOnly(thought, reply)
	if got != "Congress passed it on July 16, 1790." {
		t.Fatalf("got %q", got)
	}
}

func TestAnswerOnlyKeepsARealAnswer(t *testing.T) {
	got := AnswerOnly("The river is the Potomac.\ntalk", "Maryland and Virginia ceded the land.")
	if got != "Maryland and Virginia ceded the land." {
		t.Fatalf("got %q", got)
	}
}

func TestRouteClosedStopsOnTheFinishedLine(t *testing.T) {
	if RouteClosed("The district sits on the Potomac.\ntalk") {
		t.Fatal("the route word is still the open line")
	}
	if !RouteClosed("The district sits on the Potomac.\ntalk\n") {
		t.Fatal("talk did not close")
	}
	if RouteClosed("The district sits on the Potomac.\n") {
		t.Fatal("a sentence closed as a route")
	}
	if !RouteClosed("reason\nfact\n") {
		t.Fatal("fact did not close")
	}
	if RouteClosed("route: talk") {
		t.Fatal("a label without a newline closed")
	}
	if !RouteClosed("route: talk\n") {
		t.Fatal("a labeled route did not close")
	}
}

func TestReasoningDropsTheRoute(t *testing.T) {
	got := Reasoning("The question asks for a date.\nfact")
	if got != "The question asks for a date." {
		t.Fatalf("got %q", got)
	}
	if RouteOf("The question asks for a date.\nfact") != Fact {
		t.Fatal("route")
	}
}
