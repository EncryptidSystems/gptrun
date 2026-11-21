// Package version names the running build and notices a stale one.
//
// go build inside the checkout records the commit, its time, and whether the
// tree had uncommitted changes; String reads those back, so every build says
// what it is without flags. The release script also sets Release.
package version

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"time"
)

// Release is the release name, set by the release script with
// -ldflags "-X gptrun/internal/version.Release=0.3.0". Empty for a dev build.
var Release string

type vcs struct {
	rev   string
	when  time.Time
	dirty bool
}

func read() vcs {
	var v vcs
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return v
	}
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			v.rev = s.Value
		case "vcs.time":
			v.when, _ = time.Parse(time.RFC3339, s.Value)
		case "vcs.modified":
			v.dirty = s.Value == "true"
		}
	}
	return v
}

// Short is the release or the commit, such as "0.3.0", "c1fd988" or
// "c1fd988+dirty" (built with uncommitted changes).
func Short() string {
	v := read()
	rev := "unknown"
	if len(v.rev) >= 7 {
		rev = v.rev[:7]
	}
	if v.dirty {
		rev += "+dirty"
	}
	if Release != "" {
		return Release + " (" + rev + ")"
	}
	return rev
}

// String is the line for `gptrun version` and /api/version.
func String() string {
	s := "gptrun " + Short()
	if v := read(); !v.when.IsZero() {
		s += " " + v.when.Format("2006-01-02")
	}
	return s
}

// Stale reports that the running binary sits in its own checkout and source
// it is built from changed after it was built: the program on screen is not
// the code in the tree. The message says what to rebuild.
func Stale() (string, bool) {
	exe, err := os.Executable()
	if err != nil {
		return "", false
	}
	return staleIn(exe)
}

func staleIn(exe string) (string, bool) {
	st, err := os.Stat(exe)
	if err != nil {
		return "", false
	}
	dir := filepath.Dir(exe)
	mod, err := os.ReadFile(filepath.Join(dir, "go.mod"))
	if err != nil || !strings.HasPrefix(strings.TrimSpace(string(mod)), "module gptrun") {
		return "", false
	}
	built := st.ModTime().Add(2 * time.Second)
	newest, newestAt := "", built
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "dist", "vendor", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if strings.HasSuffix(name, "_test.go") {
			return nil
		}
		if !strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, ".wgsl") && name != "go.mod" {
			return nil
		}
		if info, err := d.Info(); err == nil && info.ModTime().After(newestAt) {
			newest, newestAt = path, info.ModTime()
		}
		return nil
	})
	if newest == "" {
		return "", false
	}
	rel, _ := filepath.Rel(dir, newest)
	return fmt.Sprintf("%s is older than %s; rebuild: go build -o %s ./cmd/gptrun",
		filepath.Base(exe), filepath.ToSlash(rel), filepath.Base(exe)), true
}
