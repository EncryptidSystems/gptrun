package hodge

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFormatSpeed(t *testing.T) {
	if formatSpeed(100, 2) != "50 tok/s" {
		t.Fatal(formatSpeed(100, 2))
	}
	if formatSpeed(5, 2) != "2.5 tok/s" {
		t.Fatal(formatSpeed(5, 2))
	}
	if formatSpeed(0, 1) != "" || formatSpeed(4, 0) != "" {
		t.Fatal("empty")
	}
}

func TestSpeedLine(t *testing.T) {
	t.Cleanup(func() { noteSpeed(0, 0) })
	noteSpeed(84, 2)
	if SpeedLine() != "42 tok/s" {
		t.Fatal(SpeedLine())
	}
}

func TestSystemPrompt(t *testing.T) {
	t.Cleanup(func() { _ = SetSystem("") })
	dir := t.TempDir()
	path := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(path, []byte("\uFEFF Be brief.\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("MODELRUN_SYSTEM", path)
	got, err := resolveSystem("Hodge")
	if err != nil || got != "Be brief." {
		t.Fatalf("%q %v", got, err)
	}
	t.Setenv("MODELRUN_SYSTEM", "Talk in short sentences.")
	got, err = resolveSystem("Hodge")
	if err != nil || got != "Talk in short sentences." {
		t.Fatalf("%q %v", got, err)
	}
	if strings.Contains(got, "https://") {
		t.Fatal("env text was read as a path")
	}
	t.Setenv("MODELRUN_SYSTEM", "Say https://example.com when asked.")
	got, err = resolveSystem("Hodge")
	if err != nil || !strings.Contains(got, "https://example.com") {
		t.Fatalf("%q %v", got, err)
	}
	t.Setenv("MODELRUN_SYSTEM", filepath.Join(dir, "missing.txt"))
	if _, err := resolveSystem("Hodge"); err == nil {
		t.Fatal("missing file")
	}
	t.Setenv("MODELRUN_SYSTEM", "From the env.")
	if err := SetSystem("From the screen."); err != nil {
		t.Fatal(err)
	}
	if System() != "From the screen." {
		t.Fatal(System())
	}
	msg, err := UseSystem("default")
	if err != nil || !strings.Contains(msg, "From the env.") {
		t.Fatalf("%q %v", msg, err)
	}
	t.Setenv("MODELRUN_SYSTEM", "")
	if err := SetSystem(""); err != nil {
		t.Fatal(err)
	}
	if System() != systemText("") {
		t.Fatal("built-in")
	}
}
