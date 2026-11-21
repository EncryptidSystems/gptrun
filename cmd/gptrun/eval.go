package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"gptrun/internal/agent"
	"gptrun/internal/generate"
	"gptrun/internal/hodge"
	"gptrun/internal/version"
)

// evalCase is one line of an eval set.
//
//	{"id":"capital-au","q":"What is the capital of Australia?","route":"fact",
//	 "want":["Canberra"],"forbid":["Sydney"]}
//
// route is the route the reply should take (fact: it led to a lookup;
// command: it led to a tool call; talk: it stood as written; recipe is
// answered as talk; "any" or empty skips it). Every entry of want must be in the answer as a
// whole word, case ignored; "a|b" takes either. No entry of forbid may be.
type evalCase struct {
	ID     string   `json:"id"`
	Q      string   `json:"q"`
	Route  string   `json:"route"`
	Want   []string `json:"want"`
	Forbid []string `json:"forbid"`
}

type evalResult struct {
	ID       string   `json:"id"`
	Q        string   `json:"q"`
	Route    string   `json:"route"`
	Took     string   `json:"took"`
	RouteOK  bool     `json:"route_ok"`
	AnswerOK bool     `json:"answer_ok"`
	Checked  bool     `json:"answer_checked"`
	Missing  []string `json:"missing,omitempty"`
	Found    []string `json:"forbidden_found,omitempty"`
	Answer   string   `json:"answer"`
	Error    string   `json:"error,omitempty"`
	Seconds  float64  `json:"seconds"`
}

type evalReport struct {
	Version string       `json:"version"`
	Model   string       `json:"model"`
	Set     string       `json:"set"`
	When    string       `json:"when"`
	Route   float64      `json:"route_accuracy"`
	Answer  float64      `json:"answer_accuracy"`
	Results []evalResult `json:"results"`
}

const evalUsage = `gptrun eval [-set FILE] [-only ID,ID] [-exec] [-min F] [-out FILE] (--config FILE | <model>)

Answers every question in the set the way gptrun run does, straight to chat,
and scores the route the reply took and the answer. Command questions only
check the route unless -exec runs them in the guest. -min fails
the run when answer accuracy is below F (0..1). The report goes to -out, by
default eval/results/<version>-<time>.json.
`

func evalCmd(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	set := fs.String("set", filepath.Join("eval", "answers.jsonl"), "eval set (JSONL)")
	config := fs.String("config", "", "checkpoint .yaml")
	only := fs.String("only", "", "comma-separated case ids")
	execute := fs.Bool("exec", false, "run command questions in the guest")
	minAcc := fs.Float64("min", 0, "fail below this answer accuracy")
	out := fs.String("out", "", "report path")
	fs.Usage = func() { fmt.Fprint(os.Stderr, evalUsage) }
	if err := fs.Parse(args); err != nil {
		return err
	}
	name := fs.Arg(0)
	if name == "" && *config == "" {
		fs.Usage()
		return fmt.Errorf("gptrun eval: a model or --config is needed")
	}
	cases, err := readEvalSet(*set)
	if err != nil {
		return err
	}
	if *only != "" {
		keep := map[string]bool{}
		for _, id := range strings.Split(*only, ",") {
			keep[strings.TrimSpace(id)] = true
		}
		var picked []evalCase
		for _, c := range cases {
			if keep[c.ID] {
				picked = append(picked, c)
			}
		}
		cases = picked
	}
	if len(cases) == 0 {
		return fmt.Errorf("gptrun eval: no cases")
	}
	target := name
	if *config != "" {
		target = *config
	}
	fmt.Fprintf(os.Stderr, "%s  %d cases  %s\n", version.String(), len(cases), target)
	if err := hodge.Prepare(target); err != nil {
		return err
	}
	opt := generate.Opt{ConfigSample: true}
	rep := evalReport{Version: version.Short(), Model: target, Set: *set, When: time.Now().Format(time.RFC3339)}
	quiet := func(string) {}
	for i, c := range cases {
		hodge.ClearInterrupt()
		hodge.Reset(target)
		r := evalResult{ID: c.ID, Q: c.Q, Route: c.Route}
		t0 := time.Now()
		var took agent.Route
		runTools := c.Route != string(agent.Command) || *execute
		r.Answer, took, err = answer(name, *config, c.Q, opt, runTools, quiet)
		r.Checked = runTools
		r.Seconds = time.Since(t0).Seconds()
		r.Took = string(took)
		if err != nil {
			r.Error = err.Error()
		}
		r.RouteOK = c.Route == "" || c.Route == "any" || c.Route == r.Took
		if r.Checked && err == nil {
			r.Missing, r.Found = scoreAnswer(r.Answer, c.Want, c.Forbid)
			r.AnswerOK = len(r.Missing) == 0 && len(r.Found) == 0
		}
		rep.Results = append(rep.Results, r)
		mark := "ok  "
		if !r.RouteOK || (r.Checked && !r.AnswerOK) {
			mark = "MISS"
		}
		detail := ""
		if !r.RouteOK {
			detail += fmt.Sprintf(" route %s, want %s;", r.Took, c.Route)
		}
		if len(r.Missing) > 0 {
			detail += " missing " + strings.Join(r.Missing, ", ") + ";"
		}
		if len(r.Found) > 0 {
			detail += " forbidden " + strings.Join(r.Found, ", ") + ";"
		}
		if r.Error != "" {
			detail += " error " + r.Error
		}
		fmt.Printf("%3d/%d  %s  %-22s %5.1fs %s\n", i+1, len(cases), mark, c.ID, r.Seconds, detail)
	}
	rep.Route, rep.Answer = evalSummary(rep.Results)
	if *out == "" {
		stamp := time.Now().Format("20060102-150405")
		*out = filepath.Join("eval", "results", strings.NewReplacer("+", "-", " ", "", "(", "", ")", "").Replace(version.Short())+"-"+stamp+".json")
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		return err
	}
	b, _ := json.MarshalIndent(rep, "", "  ")
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		return err
	}
	fmt.Println()
	printByRoute(rep.Results)
	fmt.Printf("\nroute accuracy  %5.1f%%\nanswer accuracy %5.1f%%\nreport          %s\n", 100*rep.Route, 100*rep.Answer, *out)
	if *minAcc > 0 && rep.Answer < *minAcc {
		return fmt.Errorf("answer accuracy %.1f%% is below %.1f%%", 100*rep.Answer, 100**minAcc)
	}
	return nil
}

func readEvalSet(path string) ([]evalCase, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var cases []evalCase
	seen := map[string]bool{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		var c evalCase
		dec := json.NewDecoder(strings.NewReader(line))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&c); err != nil {
			return nil, fmt.Errorf("%s:%d: %v", path, n, err)
		}
		if c.ID == "" || c.Q == "" {
			return nil, fmt.Errorf("%s:%d: id and q are needed", path, n)
		}
		if seen[c.ID] {
			return nil, fmt.Errorf("%s:%d: id %q twice", path, n, c.ID)
		}
		switch c.Route {
		case "", "any", string(agent.Fact), string(agent.Recipe), string(agent.Command), string(agent.Talk):
		default:
			return nil, fmt.Errorf("%s:%d: route %q", path, n, c.Route)
		}
		seen[c.ID] = true
		cases = append(cases, c)
	}
	return cases, sc.Err()
}

// scoreAnswer is the wanted entries not in text and the forbidden ones in it.
func scoreAnswer(text string, want, forbid []string) (missing, found []string) {
	for _, w := range want {
		if !hasWord(text, w) {
			missing = append(missing, w)
		}
	}
	for _, f := range forbid {
		if hasWord(text, f) {
			found = append(found, f)
		}
	}
	return missing, found
}

// hasWord reports that text holds one of the "|" alternatives as a whole word
// or phrase, case ignored.
func hasWord(text, alts string) bool {
	for _, alt := range strings.Split(alts, "|") {
		alt = strings.TrimSpace(alt)
		if alt == "" {
			continue
		}
		re := regexp.MustCompile(`(?i)(^|[^\pL\pN])` + regexp.QuoteMeta(alt) + `($|[^\pL\pN])`)
		if re.MatchString(text) {
			return true
		}
	}
	return false
}

func evalSummary(rs []evalResult) (route, answer float64) {
	var rn, an, ac int
	for _, r := range rs {
		if r.RouteOK {
			rn++
		}
		if r.Checked {
			ac++
			if r.AnswerOK {
				an++
			}
		}
	}
	if len(rs) > 0 {
		route = float64(rn) / float64(len(rs))
	}
	if ac > 0 {
		answer = float64(an) / float64(ac)
	}
	return route, answer
}

func printByRoute(rs []evalResult) {
	type tally struct{ n, route, checked, answer int }
	by := map[string]*tally{}
	for _, r := range rs {
		k := r.Route
		if k == "" {
			k = "any"
		}
		t := by[k]
		if t == nil {
			t = &tally{}
			by[k] = t
		}
		t.n++
		if r.RouteOK {
			t.route++
		}
		if r.Checked {
			t.checked++
			if r.AnswerOK {
				t.answer++
			}
		}
	}
	keys := make([]string, 0, len(by))
	for k := range by {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Printf("%-8s %5s %8s %8s\n", "route", "cases", "routed", "answered")
	for _, k := range keys {
		t := by[k]
		ans := "-"
		if t.checked > 0 {
			ans = fmt.Sprintf("%d/%d", t.answer, t.checked)
		}
		fmt.Printf("%-8s %5d %8s %8s\n", k, t.n, fmt.Sprintf("%d/%d", t.route, t.n), ans)
	}
}
