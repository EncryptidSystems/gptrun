// Command jsonl2st maps JSONL fields, tokenizes each mapped value, and writes
// a safetensors file. It is not hodgemix and it is not the chat runtime.
//
//	jsonl2st -in data.jsonl -tokenizer tokenizer.json -out out.safetensors -map text=body
//
// -map NAME=FIELD[,FIELD] may be repeated. FIELD is a JSON key, or a dotted
// path. Several fields are joined with a newline, and empty pieces are left
// out. Each NAME becomes NAME.ids (int32 token ids, records packed one after
// another) and NAME.row (int32 token count of each record).
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	in := flag.String("in", "", "JSONL path")
	tok := flag.String("tokenizer", "", "tokenizer.json")
	out := flag.String("out", "", "safetensors path")
	var maps mapFlag
	flag.Var(&maps, "map", "NAME=FIELD[,FIELD] (repeat for each tensor)")
	flag.Parse()
	if *in == "" || *tok == "" || *out == "" || len(maps) == 0 {
		fmt.Fprintln(os.Stderr, "jsonl2st -in data.jsonl -tokenizer tokenizer.json -out out.safetensors -map text=body")
		os.Exit(2)
	}
	rep, err := convert(*in, *tok, *out, []fieldMap(maps))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("%d rows\n", rep.Rows)
	for _, f := range rep.Fields {
		fmt.Printf("%s  %d tokens  from %s\n", f.Name, f.Tokens, f.From)
	}
	fmt.Println(*out)
}
