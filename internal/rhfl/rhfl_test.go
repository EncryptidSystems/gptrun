package rhfl

import "testing"

func TestConversationStands(t *testing.T) {
	if kind, _ := Of("hello", "Hi. What are you working on?"); kind != Stand {
		t.Fatal(kind)
	}
	if kind, _ := Of("what is the capital of France", "Paris is the capital of France."); kind != Stand {
		t.Fatal("answered")
	}
	if kind, _ := Of("Why is the cheesecake factory menu so large?", "the suitcase is too small"); kind != Stand {
		t.Fatal("short talk")
	}
	if kind, _ := Of("how to make fried chicken", "I'm not sure."); kind != Stand {
		t.Fatal("recipe")
	}
	if kind, _ := Of("write a poem about a dragon", "I'm not sure."); kind != Stand {
		t.Fatal("poem")
	}
	if kind, _ := Of("what is the capital of France", "I'm not sure France's capital is Paris, and it has been for a long time."); kind != Stand {
		t.Fatal("long hedge")
	}
}

func TestWikiCorrects(t *testing.T) {
	if kind, _ := Of("what is the capital of France", "I'm not sure."); kind != Wiki {
		t.Fatal("decline")
	}
	if kind, _ := Of("what is the capital of France", ""); kind != Wiki {
		t.Fatal("empty")
	}
	if kind, _ := Of("what is the capital of France", "fact"); kind != Wiki {
		t.Fatal("label")
	}
	if kind, _ := Of("what's the weather in Chicago", "I don't know."); kind != Wiki {
		t.Fatal("live")
	}
	if kind, _ := Of("hello", ""); kind != Stand {
		t.Fatal("empty greeting")
	}
	if kind, _ := Of("hello", "fact"); kind != Stand {
		t.Fatal("label on talk")
	}
}

func TestTool(t *testing.T) {
	kind, call := Of("list the files", `{"type":"tool_call","tool":"exec","args":{"cmd":"ls"}}`)
	if kind != Tool || call == nil || call.Cmd() != "ls" {
		t.Fatalf("%s %#v", kind, call)
	}
	kind, call = Of("read this", "Sure.\n{\"type\":\"tool_call\",\"tool\":\"gather\",\"args\":{\"url\":\"https://example.com\"}}")
	if kind != Tool || call == nil || call.Tool != "gather" {
		t.Fatalf("gather %s %#v", kind, call)
	}
}
