package hodge

import (
	"errors"
	"os"
	"strings"
	"testing"
)

// TestRecover stops the GPU process under a chat, the way a crash or a driver
// reset does. The next turn must load the model again by itself and answer as
// a fresh chat of the same history does; a stop while a reply streams must be
// ErrRestarted, keep the chat, and leave that turn out of it; a stateless call
// after a stop must just work. MODELRUN_GOLDEN=model.yaml.
func TestRecover(t *testing.T) {
	cfg := os.Getenv("MODELRUN_GOLDEN")
	if cfg == "" {
		t.Skip("MODELRUN_GOLDEN is not set")
	}
	var reloads int
	logger = func(s string) {
		if strings.HasPrefix(s, "GPU stopped") || s == "model reloaded" {
			t.Log(s)
		}
		if s == "model reloaded" {
			reloads++
		}
	}
	defer func() {
		logger = nil
		liveMu.Lock()
		if live != nil && live.gpu != nil {
			live.gpu.dev.Close()
		}
		live, liveName = nil, ""
		dropChat()
		liveMu.Unlock()
	}()
	opt := Opt{N: 24, Temp: 0, TempSet: true}
	if err := Prepare(cfg); err != nil {
		t.Fatal(err)
	}
	first, err := Continue(cfg, "Name a color.", opt, nil)
	if err != nil {
		t.Fatal(err)
	}

	// A stop between turns.
	live.gpu.dev.Kill()
	second, err := Continue(cfg, "Name another one.", opt, nil)
	if err != nil {
		t.Fatalf("turn after the stop: %v", err)
	}
	if reloads != 1 {
		t.Fatalf("reloads %d", reloads)
	}
	ref, err := Transcript(cfg, []Turn{
		{Role: "user", Content: "Name a color."}, {Role: "assistant", Content: first},
		{Role: "user", Content: "Name another one."},
	}, opt, nil)
	if err != nil {
		t.Fatal(err)
	}
	if ref != second {
		t.Fatalf("after the reload:\nchat:  %q\nfresh: %q", second, ref)
	}

	// A stop while a reply streams.
	logBefore := len(chatLog)
	var killed bool
	_, err = Continue(cfg, "And a third?", opt, func(s string) {
		if !killed {
			killed = true
			live.gpu.dev.Kill()
		}
	})
	if !errors.Is(err, ErrRestarted) {
		t.Fatalf("stop mid-reply: %v", err)
	}
	if len(chatLog) != logBefore {
		t.Fatalf("the cut-off turn went into the chat: %d -> %d", logBefore, len(chatLog))
	}
	if _, err := Continue(cfg, "And a third?", opt, nil); err != nil {
		t.Fatalf("after a restart: %v", err)
	}

	// A stateless call after a stop.
	live.gpu.dev.Kill()
	if _, err := Transcript(cfg, []Turn{{Role: "user", Content: "Say hi."}}, opt, nil); err != nil {
		t.Fatalf("stateless after a stop: %v", err)
	}
	if reloads != 3 {
		t.Fatalf("reloads %d, want 3", reloads)
	}
}
