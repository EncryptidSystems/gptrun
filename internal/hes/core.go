package hes

import "strings"

// noTime is an arrival no write has set.
const noTime = int(^uint(0) >> 1)

// phi maps a token list to a vertex when a write has no key vector. Every id
// enters the fold, so the vertex is a function of the whole list.
func phi(ids []int, n int) int {
	h := uint32(2166136261)
	for _, id := range ids {
		h ^= uint32(id)
		h *= 16777619
	}
	if n <= 1 {
		return 0
	}
	return int(h % uint32(n))
}

// Prompt is the user turn with recalled basins in front of it.
func Prompt(user, payload string) string {
	payload = strings.TrimSpace(payload)
	if payload == "" {
		return user
	}
	return "Earlier:\n" + payload + "\n\n" + user
}

// Fit returns the longest rune prefix of payload whose prompt costs at most
// budget tokens more than the bare user turn. nTok counts tokens of a user
// string. An empty result means nothing fit.
func Fit(user, payload string, budget int, nTok func(string) int) string {
	payload = strings.TrimSpace(payload)
	if payload == "" || budget <= 0 || nTok == nil {
		return ""
	}
	base := nTok(user)
	if nTok(Prompt(user, payload))-base <= budget {
		return payload
	}
	rs := []rune(payload)
	lo, hi := 1, len(rs)
	best := ""
	for lo <= hi {
		mid := (lo + hi) / 2
		cand := strings.TrimSpace(string(rs[:mid]))
		if cand == "" || nTok(Prompt(user, cand))-base > budget {
			hi = mid - 1
			continue
		}
		best = cand
		lo = mid + 1
	}
	return best
}
