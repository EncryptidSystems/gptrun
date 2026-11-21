package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"gptrun/internal/tokenizers"
)

// fieldMap is one output tensor. From are the JSON fields, in order.
type fieldMap struct {
	Name string
	From []string
}

type mapFlag []fieldMap

func (m *mapFlag) String() string { return "" }

func (m *mapFlag) Set(s string) error {
	fm, err := parseMap(s)
	if err != nil {
		return err
	}
	*m = append(*m, fm)
	return nil
}

func parseMap(s string) (fieldMap, error) {
	name, src, ok := strings.Cut(s, "=")
	name = strings.TrimSpace(name)
	if !ok || name == "" || strings.TrimSpace(src) == "" {
		return fieldMap{}, fmt.Errorf("-map %q wants NAME=FIELD[,FIELD]", s)
	}
	if strings.ContainsAny(name, " \t./\\") {
		return fieldMap{}, fmt.Errorf("-map name %q has a space or a path separator", name)
	}
	var from []string
	for _, p := range strings.Split(src, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			return fieldMap{}, fmt.Errorf("-map %s has an empty field", name)
		}
		from = append(from, p)
	}
	return fieldMap{Name: name, From: from}, nil
}

// report is what convert wrote.
type report struct {
	Rows   int
	Fields []fieldReport
}

type fieldReport struct {
	Name   string
	From   string
	Tokens int
}

type packed struct {
	ids []int32
	row []int32
}

func convert(in, tokPath, out string, maps []fieldMap) (report, error) {
	seen := map[string]bool{}
	for _, m := range maps {
		if seen[m.Name] {
			return report{}, fmt.Errorf("field %s is mapped twice", m.Name)
		}
		seen[m.Name] = true
	}
	tok, err := tokenizers.Load(tokPath)
	if err != nil {
		return report{}, err
	}
	f, err := os.Open(in)
	if err != nil {
		return report{}, err
	}
	defer f.Close()
	cols := make([]packed, len(maps))
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 32*1024*1024)
	line := 0
	rows := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" {
			continue
		}
		var obj map[string]any
		if err := json.Unmarshal([]byte(raw), &obj); err != nil {
			return report{}, fmt.Errorf("%s:%d: %v", in, line, err)
		}
		for i, m := range maps {
			text, err := joinFields(obj, m.From)
			if err != nil {
				return report{}, fmt.Errorf("%s:%d: %s: %v", in, line, m.Name, err)
			}
			ids := tok.Encode(text, true)
			cols[i].row = append(cols[i].row, int32(len(ids)))
			for _, id := range ids {
				cols[i].ids = append(cols[i].ids, int32(id))
			}
		}
		rows++
	}
	if err := sc.Err(); err != nil {
		return report{}, err
	}
	if rows == 0 {
		return report{}, fmt.Errorf("%s has no records", in)
	}
	rep := report{Rows: rows}
	var tensors []stTensor
	var meta [][2]string
	meta = append(meta, [2]string{"format", "jsonl2st"}, [2]string{"rows", strconv.Itoa(rows)})
	for i, m := range maps {
		from := strings.Join(m.From, ",")
		rep.Fields = append(rep.Fields, fieldReport{Name: m.Name, From: from, Tokens: len(cols[i].ids)})
		meta = append(meta, [2]string{"field." + m.Name, from})
		tensors = append(tensors,
			stTensor{m.Name + ".ids", "I32", []int{len(cols[i].ids)}, i32bytes(cols[i].ids)},
			stTensor{m.Name + ".row", "I32", []int{rows}, i32bytes(cols[i].row)},
		)
	}
	if err := writeST(out, meta, tensors); err != nil {
		return report{}, err
	}
	return rep, nil
}

// joinFields reads each path and joins the pieces that are not empty.
func joinFields(obj map[string]any, paths []string) (string, error) {
	var parts []string
	for _, p := range paths {
		s, err := fieldString(obj, p)
		if err != nil {
			return "", err
		}
		if s != "" {
			parts = append(parts, s)
		}
	}
	return strings.Join(parts, "\n"), nil
}

// fieldString reads a top-level key, or a dotted path when the key is absent.
func fieldString(obj map[string]any, path string) (string, error) {
	if v, ok := obj[path]; ok {
		return valueString(path, v)
	}
	cur := any(obj)
	for i, part := range strings.Split(path, ".") {
		m, ok := cur.(map[string]any)
		if !ok {
			return "", fmt.Errorf("field %s is missing", path)
		}
		v, ok := m[part]
		if !ok {
			return "", fmt.Errorf("field %s is missing", path)
		}
		if i == strings.Count(path, ".") {
			return valueString(path, v)
		}
		cur = v
	}
	return "", fmt.Errorf("field %s is missing", path)
}

func valueString(path string, v any) (string, error) {
	switch x := v.(type) {
	case string:
		return x, nil
	case nil:
		return "", nil
	case bool:
		if x {
			return "true", nil
		}
		return "false", nil
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64), nil
	case []any:
		var parts []string
		for i, el := range x {
			s, err := itemString(fmt.Sprintf("%s[%d]", path, i), el)
			if err != nil {
				return "", err
			}
			if s != "" {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "\n"), nil
	default:
		return "", fmt.Errorf("field %s is not text", path)
	}
}

func itemString(path string, v any) (string, error) {
	switch x := v.(type) {
	case string:
		return x, nil
	case map[string]any:
		content, ok := x["content"].(string)
		if !ok {
			content, ok = x["text"].(string)
		}
		if !ok {
			return "", fmt.Errorf("field %s has no content", path)
		}
		role, _ := x["role"].(string)
		if role == "" {
			return content, nil
		}
		return role + ": " + content, nil
	default:
		return valueString(path, v)
	}
}

func i32bytes(a []int32) []byte {
	b := make([]byte, 4*len(a))
	for i, v := range a {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(v))
	}
	return b
}

type stTensor struct {
	name  string
	dtype string
	shape []int
	data  []byte
}

type stInfo struct {
	Dtype string  `json:"dtype"`
	Shape []int   `json:"shape"`
	Off   []int64 `json:"data_offsets"`
}

func writeST(path string, meta [][2]string, tensors []stTensor) error {
	var data bytes.Buffer
	infos := map[string]stInfo{}
	var keys []string
	var off int64
	pad := func() {
		for off%8 != 0 {
			data.WriteByte(0)
			off++
		}
	}
	for _, t := range tensors {
		pad()
		start := off
		data.Write(t.data)
		off += int64(len(t.data))
		shape := t.shape
		if shape == nil {
			shape = []int{}
		}
		keys = append(keys, t.name)
		infos[t.name] = stInfo{Dtype: t.dtype, Shape: shape, Off: []int64{start, off}}
	}
	var hb bytes.Buffer
	hb.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			hb.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		vb, _ := json.Marshal(infos[k])
		hb.Write(kb)
		hb.WriteByte(':')
		hb.Write(vb)
	}
	hb.WriteString(`,"__metadata__":{`)
	for i, kv := range meta {
		if i > 0 {
			hb.WriteByte(',')
		}
		a, _ := json.Marshal(kv[0])
		b, _ := json.Marshal(kv[1])
		hb.Write(a)
		hb.WriteByte(':')
		hb.Write(b)
	}
	hb.WriteString("}}")
	for hb.Len()%8 != 0 {
		hb.WriteByte(' ')
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	var n [8]byte
	binary.LittleEndian.PutUint64(n[:], uint64(hb.Len()))
	if _, err := f.Write(n[:]); err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(hb.Bytes()); err != nil {
		f.Close()
		return err
	}
	if _, err := f.Write(data.Bytes()); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
