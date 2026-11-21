package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gptrun/internal/hodge"
	"gptrun/internal/version"
)

// A public item is one multiple-choice question scored the way the Eleuther
// harness scores ARC: the context ends at "Answer:", each choice is a
// continuation with a leading space, and the pick is the continuation with
// the lowest length-normalized negative log likelihood.
type pubItem struct {
	ID      string   `json:"id"`
	Context string   `json:"context"`
	Choices []string `json:"choices"`
	Gold    int      `json:"gold"`
}

type pubHit struct {
	ID       string    `json:"id"`
	Gold     int       `json:"gold"`
	Pick     int       `json:"pick"`
	PickNorm int       `json:"pick_norm"`
	OK       bool      `json:"ok"`
	OKNorm   bool      `json:"ok_norm"`
	Seconds  float64   `json:"seconds"`
	NLL      []float64 `json:"nll"`
	NTok     []int     `json:"ntok"`
}

type pubTask struct {
	Task    string   `json:"task"`
	N       int      `json:"n"`
	Acc     float64  `json:"acc"`
	AccNorm float64  `json:"acc_norm"`
	Hits    []pubHit `json:"hits"`
}

type pubReport struct {
	Version  string    `json:"version"`
	Model    string    `json:"model"`
	Protocol string    `json:"protocol"`
	When     string    `json:"when"`
	Tasks    []pubTask `json:"tasks"`
}

const benchUsage = `gptrun bench --config FILE [-task LIST] [-data DIR] [-limit N] [-out FILE]

Scores public multiple-choice sets by teacher-forced log likelihood.
0-shot. The context is "Question: ...\nAnswer:" and each choice is scored
as the continuation " <choice>". acc is the unnormalized likelihood.
acc_norm divides that likelihood by the continuation length in tokens.
The default tasks are arc-challenge, arc-easy, and openbookqa.
`

func benchCmd(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	config := fs.String("config", "", "checkpoint .yaml")
	tasks := fs.String("task", "arc-challenge,arc-easy,openbookqa", "comma-separated task names")
	data := fs.String("data", filepath.Join("eval", "public"), "directory of the public sets")
	limit := fs.Int("limit", 0, "score at most N items of each task, 0 means all")
	out := fs.String("out", "", "report path")
	fs.Usage = func() { fmt.Fprint(os.Stderr, benchUsage) }
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *config == "" {
		fs.Usage()
		return fmt.Errorf("gptrun bench: --config is needed")
	}
	var names []string
	for _, t := range strings.Split(*tasks, ",") {
		t = strings.TrimSpace(t)
		if t != "" {
			names = append(names, t)
		}
	}
	if len(names) == 0 {
		return fmt.Errorf("gptrun bench: no tasks")
	}
	fmt.Fprintf(os.Stderr, "%s  bench  %s\n", version.String(), *config)
	if err := hodge.Prepare(*config); err != nil {
		return err
	}
	rep := pubReport{
		Version:  version.Short(),
		Model:    *config,
		Protocol: "0-shot length-normalized log-likelihood; context ends at Answer:; continuation is a leading space plus the choice",
		When:     time.Now().Format(time.RFC3339),
	}
	for _, name := range names {
		items, err := loadTask(*data, name)
		if err != nil {
			return err
		}
		if *limit > 0 && *limit < len(items) {
			items = items[:*limit]
		}
		task, err := scoreTask(*config, name, items)
		if err != nil {
			return err
		}
		rep.Tasks = append(rep.Tasks, task)
		fmt.Printf("\n%s  n %d  acc %5.1f%%  acc_norm %5.1f%%\n", name, task.N, 100*task.Acc, 100*task.AccNorm)
	}
	if *out == "" {
		stamp := time.Now().Format("20060102-150405")
		*out = filepath.Join("eval", "results", "bench-"+stamp+".json")
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	fmt.Printf("report  %s\n", *out)
	return nil
}

func scoreTask(model, name string, items []pubItem) (pubTask, error) {
	task := pubTask{Task: name}
	var ok, okn int
	for i, it := range items {
		t0 := time.Now()
		hit := pubHit{ID: it.ID, Gold: it.Gold, Pick: -1, PickNorm: -1}
		best, bestN := mathInf(), mathInf()
		for c, choice := range it.Choices {
			nll, n, err := hodge.ContNLL(model, it.Context, " "+strings.TrimSpace(choice))
			if err != nil {
				return task, fmt.Errorf("%s %s: %w", name, it.ID, err)
			}
			hit.NLL = append(hit.NLL, nll)
			hit.NTok = append(hit.NTok, n)
			if nll < best {
				best = nll
				hit.Pick = c
			}
			norm := nll / float64(n)
			if norm < bestN {
				bestN = norm
				hit.PickNorm = c
			}
		}
		hit.OK = hit.Pick == it.Gold
		hit.OKNorm = hit.PickNorm == it.Gold
		hit.Seconds = time.Since(t0).Seconds()
		if hit.OK {
			ok++
		}
		if hit.OKNorm {
			okn++
		}
		task.Hits = append(task.Hits, hit)
		mark := "MISS"
		if hit.OKNorm {
			mark = "ok  "
		}
		fmt.Printf("%3d/%d  %s  %-16s %5.1fs  gold %d pick %d\n", i+1, len(items), mark, it.ID, hit.Seconds, it.Gold, hit.PickNorm)
	}
	task.N = len(items)
	if task.N > 0 {
		task.Acc = float64(ok) / float64(task.N)
		task.AccNorm = float64(okn) / float64(task.N)
	}
	return task, nil
}

func mathInf() float64 { return 1e300 }

func loadTask(root, name string) ([]pubItem, error) {
	switch name {
	case "arc-challenge":
		p, err := findNamed(root, "ARC-Challenge-Test.jsonl")
		if err != nil {
			return nil, err
		}
		return loadAI2(p)
	case "arc-easy":
		p, err := findNamed(root, "ARC-Easy-Test.jsonl")
		if err != nil {
			return nil, err
		}
		return loadAI2(p)
	case "openbookqa":
		p := filepath.Join(root, "obqa", "OpenBookQA-V1-Sep2018", "Data", "Main", "test.jsonl")
		return loadAI2(p)
	default:
		return nil, fmt.Errorf("gptrun bench: unknown task %q", name)
	}
}

func findNamed(root, name string) (string, error) {
	var hit string
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && d.Name() == name {
			hit = p
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil && hit == "" {
		return "", err
	}
	if hit == "" {
		return "", fmt.Errorf("gptrun bench: %s is not under %s", name, root)
	}
	return hit, nil
}

type ai2Choice struct {
	Text  string `json:"text"`
	Label string `json:"label"`
}

type ai2Line struct {
	ID       string `json:"id"`
	Question struct {
		Stem    string      `json:"stem"`
		Choices []ai2Choice `json:"choices"`
	} `json:"question"`
	AnswerKey string `json:"answerKey"`
}

func loadAI2(path string) ([]pubItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []pubItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var row ai2Line
		if err := json.Unmarshal([]byte(text), &row); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		labels := make([]string, len(row.Question.Choices))
		choices := make([]string, len(row.Question.Choices))
		for i, c := range row.Question.Choices {
			labels[i] = c.Label
			choices[i] = c.Text
		}
		gold := goldIndex(row.AnswerKey, labels)
		if gold < 0 {
			return nil, fmt.Errorf("%s:%d: answer %q is not in %v", path, line, row.AnswerKey, labels)
		}
		items = append(items, pubItem{
			ID:      row.ID,
			Context: "Question: " + strings.TrimSpace(row.Question.Stem) + "\nAnswer:",
			Choices: choices,
			Gold:    gold,
		})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("%s: no items", path)
	}
	return items, nil
}

func goldIndex(key string, labels []string) int {
	key = strings.TrimSpace(key)
	for i, l := range labels {
		if strings.EqualFold(l, key) {
			return i
		}
	}
	n, err := strconv.Atoi(key)
	if err == nil && n >= 1 && n <= len(labels) {
		return n - 1
	}
	return -1
}
