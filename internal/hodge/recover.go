package hodge

import (
	"errors"
	"fmt"

	"gptrun/internal/rt"
)

// A GPU process that stops (a crash, a driver reset, sleep, a hang the
// watchdog ends) used to leave the loaded model pointing at a dead pipe, and
// every turn failed until the program restarted. Now the call that finds it
// dead loads the model again, rebuilds the chat from its text, and runs once
// more.

// ErrRestarted is a reply the GPU stopped in the middle of. The model is
// loaded again and the chat kept; the message should be sent again. A reply
// already on screen is not started over silently.
var ErrRestarted = errors.New("the GPU stopped during the reply; the model was reloaded and the chat kept, send the message again")

// gone reports that m's GPU process has stopped.
func (m *Model) gone() bool {
	return m != nil && m.gpu != nil && errors.Is(m.gpu.dev.Err(), rt.ErrDead)
}

// reload drops the dead model and loads name again, with liveMu held. The
// chat comes back from its text, trimmed as a long chat is.
func reload(name string, cause error) (*Model, error) {
	logf("GPU stopped: %v; reloading the model", cause)
	if live != nil && live.gpu != nil {
		live.gpu.dev.Close()
	}
	log, was := chatLog, chatName
	live, liveName = nil, ""
	m, err := get(name)
	if err != nil {
		return nil, fmt.Errorf("the GPU stopped and the model did not load again: %w", err)
	}
	if was == name && len(log) > 0 {
		chatName = name
		if err := m.rebuildChat(log); err != nil {
			return nil, fmt.Errorf("the GPU stopped; the model is loaded again but the chat was lost: %w", err)
		}
	}
	logf("model reloaded")
	return m, nil
}

// do runs f on the loaded model, with liveMu held. When the GPU process
// stopped during f, the model is loaded again and f runs once more, unless
// again says the work cannot be repeated.
func do(name string, f func(*Model) error, again func() bool) error {
	m, err := get(name)
	if err != nil {
		return err
	}
	err = f(m)
	if err == nil || !m.gone() {
		return err
	}
	m, err = reload(name, m.gpu.dev.Err())
	if err != nil {
		return err
	}
	if again != nil && !again() {
		return ErrRestarted
	}
	return f(m)
}

// watchEmit wraps emit to remember whether any text went out. The second
// result reports that none did, so the work can run again unseen.
func watchEmit(emit func(string)) (func(string), func() bool) {
	if emit == nil {
		return nil, func() bool { return true }
	}
	sent := false
	return func(s string) {
			if s != "" {
				sent = true
			}
			emit(s)
		}, func() bool {
			return !sent
		}
}
