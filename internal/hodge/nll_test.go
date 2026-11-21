package hodge

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// The fixed sentences hodgemix eval uses for teacher-forced NLL.
var pplTexts = []string{
	"The capital of France is Paris, and the capital of Germany is Berlin.",
	"Water boils at 100 degrees Celsius at sea level.",
	"The quick brown fox jumps over the lazy dog.",
	"In 1969, Neil Armstrong became the first person to walk on the Moon.",
	"func main() { fmt.Println(\"hello, world\") }",
	"Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.",
}

func TestHodge51Perplexity(t *testing.T) {
	if os.Getenv("HODGE_PPL") == "" {
		t.Skip()
	}
	const cfg = `C:\Users\grego\models\dev\hodge-5.1\hodge-5.1.bnz.yaml`
	SetLog(func(s string) { fmt.Fprintln(os.Stderr, s) })
	if err := Prepare(cfg); err != nil {
		t.Fatal(err)
	}
	var tot float64
	var tokens int
	for _, text := range pplTexts {
		sum, n, err := live.textNLL(text)
		if err != nil {
			t.Fatal(err)
		}
		mean := sum / float64(n)
		tot += sum
		tokens += n
		t.Logf("nll %.3f  tokens %d  %q", mean, n, text)
	}
	mean := tot / float64(tokens)
	t.Logf("mean NLL %.3f nats/token over %d tokens (ppl %.1f)", mean, tokens, math.Exp(mean))
}
