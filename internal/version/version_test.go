package version

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStale(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, when time.Time) {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("module gptrun\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, when, when); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Now()
	write("go.mod", now.Add(-time.Hour))
	write("internal/x/x.go", now.Add(-time.Hour))
	write("gptrun.exe", now.Add(-30*time.Minute))
	exe := filepath.Join(dir, "gptrun.exe")
	if msg, stale := staleIn(exe); stale {
		t.Fatalf("fresh build called stale: %s", msg)
	}
	// A test file does not make the binary stale; a source file does.
	write("internal/x/x_test.go", now)
	if _, stale := staleIn(exe); stale {
		t.Fatal("test file made it stale")
	}
	write("internal/hodge/kernels/k.wgsl", now)
	msg, stale := staleIn(exe)
	if !stale || !strings.Contains(msg, "k.wgsl") || !strings.Contains(msg, "go build") {
		t.Fatalf("%v %q", stale, msg)
	}
	// Outside a gptrun checkout there is nothing to compare.
	if _, stale := staleIn(filepath.Join(t.TempDir(), "gptrun.exe")); stale {
		t.Fatal("no checkout")
	}
}

func TestShort(t *testing.T) {
	if Short() == "" {
		t.Fatal("empty")
	}
	Release = "9.9.9"
	defer func() { Release = "" }()
	if !strings.HasPrefix(Short(), "9.9.9 (") {
		t.Fatal(Short())
	}
}
