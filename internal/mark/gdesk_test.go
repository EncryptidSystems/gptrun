package mark

import (
	"os"
	"strings"
	"testing"
)

func TestGDeskPage(t *testing.T) {
	html := string(gdeskHTML)
	for _, want := range []string{"g(w)", "Gemini", "ENCRYPTID", "/api/g", "/api/word"} {
		if !strings.Contains(html, want) {
			t.Fatalf("page missing %q", want)
		}
	}
}

func TestOpenGDesk(t *testing.T) {
	vocab := `C:\Users\grego\models\prod\2b4t\vocab.float.safetensors`
	tok := `C:\Users\grego\models\prod\2b4t\tokenizer.json`
	corpus := `C:\Users\grego\watermarks`
	if _, err := os.Stat(vocab); err != nil {
		t.Skip(err)
	}
	d, err := OpenGDesk(vocab, tok, corpus)
	if err != nil {
		t.Fatal(err)
	}
	s := d.Snap()
	if s.PoemN < 100 || s.PoemMean < s.BaseMean+0.15 {
		t.Fatalf("poem mean %.3f (n %d) base mean %.3f", s.PoemMean, s.PoemN, s.BaseMean)
	}
	if s.Agreed < 40 || len(s.Mask) != s.Cells {
		t.Fatalf("agreed %d mask %d cells %d", s.Agreed, len(s.Mask), s.Cells)
	}
	g, ok := d.ScoreWord("willow")
	if !ok || g <= 0 {
		t.Fatalf("willow g %.3f ok %v", g, ok)
	}
	if _, ok := d.ScoreWord("a"); ok {
		t.Fatal("a single letter scored")
	}
}
