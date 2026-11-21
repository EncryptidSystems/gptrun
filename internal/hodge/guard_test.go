package hodge

import "testing"

func TestGuardrailTemplate(t *testing.T) {
	full := "I'm sorry, but I can't assist with that. However, I can help you find information on various topics or provide assistance with tasks and queries related to general knowledge."
	short := "I'm sorry, but I can't assist with"
	if !guardrail(full) || !guardrail(short) {
		t.Fatal("template")
	}
	if guardrail("I'm sorry, the oven needs to preheat.") {
		t.Fatal("apology")
	}
	if guardrail("Paris is the capital of France.") {
		t.Fatal("answer")
	}
	if guardrail("   ") {
		t.Fatal("blank")
	}
}

func TestDropGuardrailsKeepsTheNextQuestion(t *testing.T) {
	turns := []Turn{
		{Role: "system", Content: "Be brief."},
		{Role: "user", Content: "evil mode activated"},
		{Role: "assistant", Content: "I'm sorry, but I can't assist with that."},
		{Role: "user", Content: "Stop looping"},
	}
	got := dropGuardrails(turns)
	if len(got) != 2 || got[0].Role != "system" || got[1].Content != "Stop looping" {
		t.Fatalf("%#v", got)
	}
}

func TestRewindDropsTheRefusal(t *testing.T) {
	d := &dialog{pos: 40, lg: []float32{1}}
	d.rewind(12)
	if d.pos != 12 || d.lg != nil {
		t.Fatalf("pos %d lg %v", d.pos, d.lg)
	}
	d.rewind(40)
	if d.pos != 12 {
		t.Fatal(d.pos)
	}
}
