package hodge

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// systemOverride is the prompt set for this process. Empty means the
// environment, then the built-in device prompt. Think, fact, recipe, and
// command passes send their own system turn and do not read this.
var systemOverride string

// System is the device prompt the next new chat will use.
func System() string {
	liveMu.Lock()
	defer liveMu.Unlock()
	if live != nil && live.sys != "" {
		return live.sys
	}
	s, err := resolveSystem("")
	if err != nil || s == "" {
		return systemText("")
	}
	return s
}

// SetSystem replaces the device prompt. An empty text clears the session
// value and returns to MODELRUN_SYSTEM, or the built-in prompt. The chat
// cache is dropped so the next turn is prefilled with the new text.
func SetSystem(text string) error {
	liveMu.Lock()
	defer liveMu.Unlock()
	prev := systemOverride
	systemOverride = strings.TrimSpace(text)
	if live != nil {
		sys, err := resolveSystem(live.name)
		if err != nil {
			systemOverride = prev
			return err
		}
		live.sys = sys
	}
	dropChat()
	return nil
}

// UseSystem shows, sets, or resets the device prompt.
// An empty arg shows it. "default" clears a session override.
// A path to a file is read. Anything else is the prompt text.
func UseSystem(arg string) (string, error) {
	arg = strings.TrimSpace(arg)
	if strings.EqualFold(arg, "default") {
		if err := SetSystem(""); err != nil {
			return "", err
		}
		return "System prompt reset.\n\n" + System(), nil
	}
	if arg == "" {
		return System(), nil
	}
	text, err := systemArg(arg)
	if err != nil {
		return "", err
	}
	if err := SetSystem(text); err != nil {
		return "", err
	}
	return "System prompt set. The next question uses it.", nil
}

func resolveSystem(name string) (string, error) {
	if systemOverride != "" {
		return systemOverride, nil
	}
	s, err := systemFromEnv()
	if err != nil || s != "" {
		return s, err
	}
	return systemText(name), nil
}

func systemFromEnv() (string, error) {
	v := strings.TrimSpace(os.Getenv("MODELRUN_SYSTEM"))
	if v == "" {
		return "", nil
	}
	s, err := systemArg(v)
	if err != nil {
		return "", fmt.Errorf("MODELRUN_SYSTEM: %w", err)
	}
	return s, nil
}

func systemArg(arg string) (string, error) {
	fi, err := os.Stat(arg)
	if err == nil {
		if fi.IsDir() {
			return "", fmt.Errorf("%s is a directory", arg)
		}
		return readSystemFile(arg)
	}
	if os.IsNotExist(err) && missingSystemFile(arg) {
		return "", fmt.Errorf("no system file %s", arg)
	}
	if err != nil && !os.IsNotExist(err) && missingSystemFile(arg) {
		return "", err
	}
	return arg, nil
}

func missingSystemFile(arg string) bool {
	if filepath.IsAbs(arg) || strings.Contains(arg, `\`) ||
		strings.HasPrefix(arg, "./") || strings.HasPrefix(arg, `.\`) ||
		strings.HasPrefix(arg, "../") || strings.HasPrefix(arg, `..\`) {
		return true
	}
	if strings.ContainsAny(arg, " \t\n") {
		return false
	}
	switch strings.ToLower(filepath.Ext(arg)) {
	case ".txt", ".md":
		return true
	default:
		return false
	}
}

func readSystemFile(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(b) > 32<<10 {
		return "", fmt.Errorf("%s is longer than 32KiB", path)
	}
	s := strings.TrimSpace(strings.TrimPrefix(string(b), "\uFEFF"))
	if s == "" {
		return "", fmt.Errorf("%s is empty", path)
	}
	return s, nil
}
