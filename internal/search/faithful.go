package search

import (
	"strings"
	"unicode"
	"unicode/utf8"

	"gptrun/internal/agent"
)

// Faithful is answer without the sentences that name something the notes in
// prompt and the question do not. A small model asked to write from thin
// notes invents the rest: a river, a television station, a year. Those are
// names and numbers, and none of them is in the notes, so their sentences go.
// A heading line goes too. The empty string means no sentence stood on the
// notes; the caller shows the notes instead.
//
// A name is a capitalized word after a sentence's first word, or a word with
// a digit. The first word is capitalized anyway; it is not read as a name.
func Faithful(answer, prompt, question string) string {
	support := wordsOf(Brief(prompt) + " " + question)
	body := strings.TrimSpace(answer)
	if i := strings.Index(body, "\nSources:"); i >= 0 {
		body = body[:i]
	}
	var paras []string
	for _, para := range strings.Split(body, "\n") {
		line := strings.TrimSpace(para)
		if line == "" || heading(line) {
			continue
		}
		// After an invented sentence the paragraph goes on about the
		// invention ("The river has played a role…"), so the rest goes too.
		var kept []string
		for _, s := range splitSentences(line) {
			s = strings.TrimSpace(s)
			if len(strings.Fields(s)) < 3 {
				continue
			}
			if !standsOn(s, support) {
				break
			}
			kept = append(kept, s)
		}
		if len(kept) > 0 {
			paras = append(paras, strings.Join(kept, " "))
		}
	}
	return strings.Join(paras, "\n\n")
}

// Settle is the reply to show for a looked-up question: answer's sentences
// that stand on the notes in prompt, or the notes themselves when none do or
// what is left does not answer the question. The empty string means neither
// carries the question's name.
func Settle(question, prompt, answer string) string {
	text := Faithful(answer, prompt, question)
	// Every name left is in the notes, so a direct answer that does not
	// repeat the subject ("James Rouse founded it in 1967.") stands; it only
	// has to share a word with the question.
	onTopic := CarriesName(question, text) || scoreSentence(text, queryWords(question)) > 0
	if text == "" || agent.RouteOnly(text) || OffSubject(question, text) || !onTopic {
		notes := Brief(prompt)
		if !CarriesName(question, notes) {
			return ""
		}
		return notes
	}
	return text
}

func heading(line string) bool {
	if strings.HasPrefix(line, "#") {
		return true
	}
	lower := strings.ToLower(line)
	return strings.HasPrefix(lower, "title:") || strings.HasPrefix(lower, "sources:")
}

// standsOn reports that every name in sentence is a word of support.
func standsOn(sentence string, support []string) bool {
	for i, raw := range strings.Fields(sentence) {
		w := strings.TrimFunc(raw, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
		if w == "" {
			continue
		}
		first, _ := utf8.DecodeRuneInString(w)
		digit := strings.IndexFunc(w, unicode.IsDigit) >= 0
		name := i > 0 && unicode.IsUpper(first)
		if !digit && !name {
			continue
		}
		for _, part := range wordsOf(w) {
			if len(part) < 2 && !digit {
				continue
			}
			if !tokenHas(support, part) {
				return false
			}
		}
	}
	return true
}
