package hodge

import (
	"errors"
	"fmt"
	"strings"
)

// A conversation longer than the context window used to fail every turn with
// "context full" until the program restarted. The cache cannot drop its oldest
// rows (each row was computed at its position), so the chat keeps its turns
// as text too, and when the next turn would not fit it is rebuilt from the
// system prompt and the latest turns. A rebuild reads those turns again, so it
// keeps a quarter of the window: the next one is a long way off. Turns that
// leave the window stay in the holonomy–eikonal field and can be read back
// into the tokens the new turn still has free.

// replyRoom is the room a turn leaves for the reply.
const replyRoom = 512

// ErrTooLong is a single message that does not fit the context window.
var ErrTooLong = errors.New("the message is too long for the context window")

// chatLog is the TUI chat as text, in step with chat.
var chatLog []Turn

func dropChat() {
	chat, chatName, chatLog = nil, "", nil
	chatHES = nil // a -hes state file reloads it on the next turn
	promptedUser = ""
	sightings = nil
}

// render is the token ids of turns under sys, as a fresh transcript.
func (m *Model) render(sys string, turns []Turn) ([]int, error) {
	var ids []int
	first := true
	for _, t := range turns {
		switch strings.ToLower(t.Role) {
		case "system":
		case "user":
			use := ""
			if first {
				use = sys
			}
			ids = append(ids, turnIDs(m.tok, m.kind, m.bos, use, t.Content, first)...)
			first = false
		case "assistant":
			ids = append(ids, m.pieces(t.Content)...)
			if end := m.stopPiece(); end != "" {
				ids = append(ids, m.pieces(end)...)
			}
		default:
			return nil, fmt.Errorf("role %q", t.Role)
		}
	}
	if first {
		return nil, fmt.Errorf("empty prompt")
	}
	return ids, nil
}

// ctxLen is the context window in tokens.
func (m *Model) ctxLen() int {
	if m.gpu != nil {
		return m.gpu.ctx
	}
	return 4096
}

func room(opt Opt) int {
	if opt.N > 0 && opt.N < replyRoom {
		return opt.N
	}
	return replyRoom
}

// fitTurns drops the oldest turns until the rest renders to at most budget tokens,
// and to at most the window less reserve. It always keeps the last user turn
// and what follows it. dropped counts the turns left out.
func (m *Model) fitTurns(sys string, turns []Turn, budget, reserve int) (ids []int, kept []Turn, dropped int, err error) {
	window := m.ctxLen() - reserve
	budget = min(budget, window)
	lastUser := -1
	for i, t := range turns {
		if strings.EqualFold(t.Role, "user") {
			lastUser = i
		}
	}
	if lastUser < 0 {
		return nil, nil, 0, fmt.Errorf("empty prompt")
	}
	start := 0
	for {
		// The kept turns start at a user turn.
		for start < lastUser && !strings.EqualFold(turns[start].Role, "user") {
			start++
		}
		ids, err = m.render(sys, turns[start:])
		if err != nil {
			return nil, nil, 0, err
		}
		if len(ids) <= budget || start >= lastUser {
			break
		}
		start++
	}
	if len(ids) > window {
		return nil, nil, 0, fmt.Errorf("%w (%d tokens; the window holds %d with room for the reply)", ErrTooLong, len(ids), window)
	}
	for _, t := range turns[:start] {
		if !strings.EqualFold(t.Role, "system") {
			dropped++
		}
	}
	return ids, turns[start:], dropped, nil
}

// rebuildChat starts the chat over from the latest turns of log that fit a
// quarter of the window, leaving the next turn to the caller. An empty log is
// a fresh chat. The cache keeps the system prompt's rows, so only the kept
// turns are read again.
func (m *Model) rebuildChat(log []Turn) error {
	d := &dialog{}
	if len(log) == 0 {
		chat, chatLog = d, nil
		logf("chat trimmed  every old turn dropped")
		return nil
	}
	ids, kept, dropped, err := m.fitTurns(systemFor(m), log, m.ctxLen()/4, replyRoom)
	if err != nil {
		return err
	}
	logf("chat trimmed  %d old turns dropped, %d kept", dropped, len(kept))
	if err := m.prefill(d, ids); err != nil {
		return err
	}
	chat, chatLog = d, append([]Turn(nil), kept...)
	return nil
}

// nextTurn is the ids of user's turn in the chat, after rebuilding the chat
// when they and reserve would not fit. It fails only for a message that does
// not fit an empty chat.
func (m *Model) nextTurn(user string, reserve int) ([]int, error) {
	return m.nextTurnAt(user, reserve, true)
}

// nextTurnAt is nextTurn. recall reads one forgotten basin into the leftover
// tokens. A stored answer does not: the text being saved was already written.
func (m *Model) nextTurnAt(user string, reserve int, recall bool) ([]int, error) {
	// A message that would not fit even an empty chat is refused before the
	// chat is touched: a long paste does not cost the conversation. The check
	// uses the text the person sent. A recalled basin is added only when the
	// leftover tokens can hold it.
	alone := turnIDs(m.tok, m.kind, m.bos, systemFor(m), user, true)
	if len(alone)+reserve > m.ctxLen() {
		// Longer than the window: the head goes into the chat's
		// holonomy–eikonal field, the tail stays as the turn.
		if short, ok := m.overflowUser(user, reserve); ok {
			user = short
			alone = turnIDs(m.tok, m.kind, m.bos, systemFor(m), user, true)
		}
	}
	if len(alone)+reserve > m.ctxLen() {
		return nil, fmt.Errorf("%w (%d tokens; the window holds %d with room for the reply)", ErrTooLong, len(alone), m.ctxLen()-reserve)
	}
	try := func() ([]int, bool) {
		text := user
		if recall {
			text = m.earlierHES(user, reserve)
		}
		ids := m.turnFor(text)
		if chat.pos+len(ids)+reserve <= m.ctxLen() {
			if recall {
				promptedUser = text
			}
			return ids, true
		}
		return nil, false
	}
	if ids, ok := try(); ok {
		return ids, nil
	}
	for _, log := range [][]Turn{chatLog, nil} {
		if err := m.rebuildChat(log); err != nil && !errors.Is(err, ErrTooLong) {
			return nil, err
		}
		if ids, ok := try(); ok {
			return ids, nil
		}
	}
	return nil, fmt.Errorf("%w (%d tokens; the window holds %d with room for the reply)", ErrTooLong, len(alone), m.ctxLen()-reserve)
}

// turnFor is user's turn in the chat as it stands: the first carries the
// system prompt.
func (m *Model) turnFor(user string) []int {
	if chat.pos == 0 {
		return turnIDs(m.tok, m.kind, m.bos, systemFor(m), user, true)
	}
	return turnIDs(m.tok, m.kind, nil, "", user, false)
}
