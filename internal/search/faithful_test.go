package search

import (
	"strings"
	"testing"
)

// The reply that started this, written from one line of notes.
func TestFaithfulDropsInventedNames(t *testing.T) {
	prompt := "The lookup is already done.\nNotes:\nColumbia, Maryland: Columbia is a census-designated place in Howard County, Maryland. " +
		"It was planned by James Rouse and founded in 1967.\nQuestion: Tell me about the history and culture of Columbia, Maryland"
	q := "Tell me about the history and culture of Columbia, Maryland"
	answer := "Title: The History and Culture of Columbia, Maryland\n\n" +
		"Columbia, Maryland is a planned community in Howard County. It was founded in 1967 by James Rouse.\n\n" +
		"The city's name is derived from its geographical location, situated on the banks of the Patuxka River, which flows through the city. " +
		"The river has played a significant role in shaping the landscape.\n\n" +
		"One of the most famous landmarks in Columbia is the \"Flying High\" television station, which operated from 1978 until 1980. " +
		"U.S. Route 29 passes through the city.\n\n" +
		"Sources:\n1. [Columbia, Maryland](https://en.wikipedia.org/wiki/Columbia,_Maryland)"
	got := Faithful(answer, prompt, q)
	for _, bad := range []string{"Title", "Patuxka", "The river has played", "Flying High", "1978", "Route 29", "Sources"} {
		if strings.Contains(got, bad) {
			t.Errorf("kept %q:\n%s", bad, got)
		}
	}
	for _, good := range []string{"planned community in Howard County", "founded in 1967 by James Rouse"} {
		if !strings.Contains(got, good) {
			t.Errorf("dropped %q:\n%s", good, got)
		}
	}
}

func TestFaithfulNothingStands(t *testing.T) {
	prompt := "Notes:\nPage: Columbia is in Maryland.\nQuestion: Who founded Columbia?"
	if got := Faithful("It was founded by Walt Disney in 1955.", prompt, "Who founded Columbia?"); got != "" {
		t.Fatalf("%q", got)
	}
}

func TestAnswerSentencesReachPastTheLead(t *testing.T) {
	page := "Columbia is a planned community in Howard County, Maryland. It has ten villages and a downtown. " +
		"Columbia is part of the Baltimore metropolitan area and has many parks. " +
		"The community was conceived by real estate developer James W. Rouse, who founded it in 1967. " +
		"The Little Patuxent River runs through the town and feeds Lake Kittamaqundi."
	got := answerSentences(page, "Who founded Columbia, Maryland?", 1)
	if len(got) != 1 || !strings.Contains(got[0], "Rouse") {
		t.Fatalf("founder: %q", got)
	}
	got = answerSentences(page, "What river runs through Columbia, Maryland?", 1)
	if len(got) != 1 || !strings.Contains(got[0], "Little Patuxent") {
		t.Fatalf("river: %q", got)
	}
}

func TestSplitKeepsInitials(t *testing.T) {
	got := splitSentences("Columbia was founded in 1967 by James W. Rouse. It is near Washington, D.C. and Baltimore. The U.S. Census counts it. Done.")
	want := []string{"Columbia was founded in 1967 by James W. Rouse.", "It is near Washington, D.C. and Baltimore.", "The U.S. Census counts it.", "Done."}
	if len(got) != len(want) {
		t.Fatalf("%q", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%d: %q", i, got[i])
		}
	}
}
