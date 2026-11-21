package rag

import (
	"testing"
)

func TestAugmentWithoutStore(t *testing.T) {
	t.Setenv("EMBEDDB_NOTES", "1")
	t.Setenv("EMBEDDB_CORPUS", t.TempDir()+`\missing-corpus`)
	t.Setenv("EMBEDDB_STORE", t.TempDir()+`\missing.edb`)
	if got := Augment("what is hodge-5"); got != "what is hodge-5" {
		t.Fatalf("%q", got)
	}
	if got := Augment("/run echo hi"); got != "/run echo hi" {
		t.Fatalf("%q", got)
	}
}

func TestAugmentNotesOff(t *testing.T) {
	t.Setenv("EMBEDDB_NOTES", "")
	if got := Augment("suspicious php on the web server"); got != "suspicious php on the web server" {
		t.Fatalf("%q", got)
	}
}
