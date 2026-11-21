package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"gptrun/internal/tokenizers"
)

const llama3Split = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

func writeTok(t *testing.T, dir string) string {
	t.Helper()
	vocab := map[string]int{}
	n := 0
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
			r = rune(256 + n)
			n++
		}
		vocab[string(r)] = b
	}
	root := map[string]any{
		"model": map[string]any{"type": "BPE", "vocab": vocab, "merges": []any{}},
		"pre_tokenizer": map[string]any{
			"type": "Sequence",
			"pretokenizers": []any{
				map[string]any{"type": "Split", "pattern": map[string]any{"Regex": llama3Split}},
				map[string]any{"type": "ByteLevel", "use_regex": false},
			},
		},
	}
	b, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "tokenizer.json")
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestConvertMapsAndTokenizes(t *testing.T) {
	dir := t.TempDir()
	tokPath := writeTok(t, dir)
	tok, err := tokenizers.Load(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	in := filepath.Join(dir, "in.jsonl")
	body := "" +
		"{\"headline\":\"Ab\",\"body\":\"Ab Cd\",\"note\":{\"city\":\"\"},\"turns\":[]}\n" +
		"\n" +
		"{\"headline\":\"Z\",\"body\":\"\",\"note\":{\"city\":\"DC\"},\"turns\":[{\"role\":\"user\",\"content\":\"Hi\"}]}\n"
	if err := os.WriteFile(in, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out.safetensors")
	maps := []fieldMap{
		{Name: "title", From: []string{"headline"}},
		{Name: "text", From: []string{"body", "note.city", "turns"}},
	}
	rep, err := convert(in, tokPath, out, maps)
	if err != nil {
		t.Fatal(err)
	}
	if rep.Rows != 2 {
		t.Fatalf("rows %d", rep.Rows)
	}
	got := readI32(t, out)
	wantTitle := append(encode(tok, "Ab"), encode(tok, "Z")...)
	wantText := append(encode(tok, "Ab Cd"), encode(tok, "DC\nuser: Hi")...)
	if !sameI32(got["title.ids"], wantTitle) {
		t.Fatalf("title.ids %v want %v", got["title.ids"], wantTitle)
	}
	if !sameI32(got["text.ids"], wantText) {
		t.Fatalf("text.ids %v want %v", got["text.ids"], wantText)
	}
	if !sameI32(got["title.row"], []int32{int32(len(encode(tok, "Ab"))), int32(len(encode(tok, "Z")))}) {
		t.Fatalf("title.row %v", got["title.row"])
	}
	if !sameI32(got["text.row"], []int32{int32(len(encode(tok, "Ab Cd"))), int32(len(encode(tok, "DC\nuser: Hi")))}) {
		t.Fatalf("text.row %v", got["text.row"])
	}
	if got["title.ids"] == nil || len(got["text.ids"]) != len(wantText) {
		t.Fatalf("tensors %v", len(got))
	}
}

func TestConvertMissingField(t *testing.T) {
	dir := t.TempDir()
	tokPath := writeTok(t, dir)
	in := filepath.Join(dir, "in.jsonl")
	if err := os.WriteFile(in, []byte("{\"a\":\"Ab\"}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := convert(in, tokPath, filepath.Join(dir, "out.safetensors"), []fieldMap{{Name: "text", From: []string{"body"}}})
	if err == nil {
		t.Fatal("missing field was accepted")
	}
}

func encode(tok *tokenizers.Tokenizer, text string) []int32 {
	ids := tok.Encode(text, true)
	o := make([]int32, len(ids))
	for i, id := range ids {
		o[i] = int32(id)
	}
	return o
}

func sameI32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func readI32(t *testing.T, path string) map[string][]int32 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	n := int(binary.LittleEndian.Uint64(b[:8]))
	var header map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &header); err != nil {
		t.Fatal(err)
	}
	base := 8 + n
	out := map[string][]int32{}
	for k, raw := range header {
		if k == "__metadata__" {
			continue
		}
		var info stInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			t.Fatal(err)
		}
		if info.Dtype != "I32" {
			t.Fatalf("%s dtype %s", k, info.Dtype)
		}
		chunk := b[base+int(info.Off[0]) : base+int(info.Off[1])]
		if len(chunk)%4 != 0 {
			t.Fatalf("%s len %d", k, len(chunk))
		}
		ids := make([]int32, len(chunk)/4)
		for i := range ids {
			ids[i] = int32(binary.LittleEndian.Uint32(chunk[4*i:]))
		}
		out[k] = ids
		if info.Off[0]%8 != 0 {
			t.Fatalf("%s offset %d", k, info.Off[0])
		}
	}
	meta := map[string]string{}
	if err := json.Unmarshal(header["__metadata__"], &meta); err != nil {
		t.Fatal(err)
	}
	if meta["format"] != "jsonl2st" || meta["field.title"] != "headline" || meta["rows"] != "2" {
		t.Fatalf("meta %v", meta)
	}
	if bytes.Contains(b[8:8+n], []byte("hodgemix")) {
		t.Fatal("header names hodgemix")
	}
	return out
}
