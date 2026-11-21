package hodge

import "strings"

// A memory file (-memory) is passages separated by a line "---". withMemoryHES
// (hesrun.go) writes each passage into a library field and reads the field for
// the user turn and each of its sentences.

func splitPassages(text string) []string {
	var out []string
	var cur []string
	flush := func() {
		p := strings.TrimSpace(strings.Join(cur, "\n"))
		if p != "" {
			out = append(out, p)
		}
		cur = cur[:0]
	}
	for _, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		if strings.TrimSpace(line) == "---" {
			flush()
			continue
		}
		cur = append(cur, line)
	}
	flush()
	return out
}

// queryViews is the whole user turn, then each sentence and line of it.
func queryViews(user string) []string {
	views := []string{user}
	seen := map[string]bool{strings.TrimSpace(user): true}
	cut := strings.NewReplacer(". ", ".\n", "? ", "?\n", "! ", "!\n", "$$", "\n")
	for _, s := range strings.Split(cut.Replace(user), "\n") {
		s = strings.TrimSpace(s)
		if len(s) < 8 || seen[s] {
			continue
		}
		seen[s] = true
		views = append(views, s)
	}
	return views
}
