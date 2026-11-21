package hodge

import (
	"fmt"
	"strings"
	"sync"

	"gptrun/internal/chatfmt"
)

// The chat format a run asks for. A checkpoint can carry the role template in
// tokenizer_config and still be a Llama 3 model in its tokens (Hodge-5o); the
// default stays the file's template, and UseChat picks another one it has.
var (
	chatMu   sync.Mutex
	chatPick chatfmt.Kind
)

// UseChat sets the chat format for the models this process loads: role,
// llama3, gemma, qwen, harmony, or plain. "" or "default" is the file's own.
func UseChat(name string) error {
	k := chatfmt.Kind(strings.ToLower(strings.TrimSpace(name)))
	switch k {
	case "", "default":
		k = ""
	case chatfmt.Role, chatfmt.Llama3, chatfmt.Gemma, chatfmt.Qwen, chatfmt.Harmony, chatfmt.Plain:
	default:
		return fmt.Errorf("chat format %q: use role, llama3, gemma, qwen, harmony, or plain", name)
	}
	chatMu.Lock()
	chatPick = k
	chatMu.Unlock()
	return nil
}

func pickedChat() chatfmt.Kind {
	chatMu.Lock()
	defer chatMu.Unlock()
	return chatPick
}

// pickFormat is the asked-for format when the tokenizer has its tokens.
func pickFormat(have []chatfmt.Kind) (chatfmt.Kind, error) {
	want := pickedChat()
	if want == "" {
		return "", nil
	}
	for _, k := range have {
		if k == want {
			return k, nil
		}
	}
	if want == chatfmt.Plain {
		return want, nil
	}
	return "", fmt.Errorf("chat format %s: this tokenizer does not have its turn tokens", want)
}
