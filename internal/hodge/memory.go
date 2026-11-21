package hodge

import "strings"

// The chat's memory is the holonomy–eikonal state in hesrun.go. These are the
// chat-log helpers its reads use: a basin already written into the visible
// chat is skipped until the turn that carries it is trimmed.

// promptedUser is the user text last given to the model.
var promptedUser string

// A sighting is a basin already written into the chat log. It stays skipped
// until the turn that carries it is trimmed, and then the field can return it.
type sighting struct {
	payload string
	fitted  string
}

var sightings []sighting

// visible reports whether payload is already in the chat the model can see:
// as its own turn, as the question under an earlier basin, or as that basin.
func visible(payload string, log []Turn) bool {
	if payload == "" {
		return true
	}
	block := "Earlier:\n" + payload + "\n\n"
	tail := "\n\n" + payload
	for _, t := range log {
		if t.Content == payload || strings.Contains(t.Content, block) || strings.HasSuffix(t.Content, tail) {
			return true
		}
	}
	return false
}

func pruneSightings() {
	kept := sightings[:0]
	for _, s := range sightings {
		block := "Earlier:\n" + s.fitted + "\n\n"
		for _, t := range chatLog {
			if strings.Contains(t.Content, block) {
				kept = append(kept, s)
				break
			}
		}
	}
	sightings = kept
}

func skipPayload(payload, user string) bool {
	if payload == "" || payload == user || visible(payload, chatLog) {
		return true
	}
	pruneSightings()
	for _, s := range sightings {
		if s.payload == payload {
			return true
		}
	}
	return false
}
