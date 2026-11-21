# ModelRun

ModelRun is the Hodge-Kuramoto runtime. A packed linear runs as `hodgelinear.wgsl` on Dawn, limbs included. A dense bf16 linear runs as `dense.wgsl`. LUT kernels are the WGSL files Dawn loads as written. The `model_type` stamp in `config.json` is not the runtime name.

A checkpoint is a safetensors file or a BNZ4 file, plus its tokenizer and `config.json`. Chat text is rendered from that tokenizer's own pieces: role, llama3, gemma, qwen, harmony, or plain. `gptrun list` shows the Dawn clock and the WGSL kernels. A checkpoint is opened with `--config`.

Commands the model asks for run in the TunnelTug guest. They do not run on this machine.

## Build

Go 1.24. The module replaces `hodgemix` with a checkout at `../hodgemix-go` and `embeddb` with `../EmbedDB`.

Every build knows what it is: `gptrun version` prints the commit it was built from, `+dirty` when the tree had uncommitted changes. The chat screen shows it next to Ready, and warns when `gptrun.exe` is older than the source beside it.

```
go build -o gptrun.exe ./cmd/gptrun
```

The same module builds `cochainkey`, `cochainimg`, and `jsonl2st`. `jsonl2st` is not hodgemix. It maps named JSONL fields, tokenizes them, and writes safetensors (`NAME.ids` and `NAME.row`):

```
go build -o cochainkey.exe ./cmd/cochainkey
go build -o cochainimg.exe ./cmd/cochainimg
go build -o jsonl2st.exe ./cmd/jsonl2st
```

```
jsonl2st -in data.jsonl -tokenizer tokenizer.json -out out.safetensors -map text=body
```

Hodge dispatches need `dawnrt.exe`. The resident Dawn host is `dawnhost.exe`. The Dawn DLLs sit next to each of those programs. LUT runs and the clock need WASMApollo's `dawn.exe`.

`dawnrt.exe` is built from `dawnrt/rt.cc`. The build needs the Visual Studio C++ tools and the Dawn chromium/7187 prebuilt that `WASMApollo\dawn\build.bat` fetches. `DAWN_PREBUILT` names another copy. ModelRun refuses a `dawnrt.exe` older than its own `rt.cc`.

```
dawnrt\build.bat
```

### Release

```
$env:MODELRUN_GOLDEN = 'C:\path\to\model.bnz.yaml'
pwsh .\release.ps1 -Release 0.3.0
```

The script refuses uncommitted changes in ModelRun or `../hodgemix-go` (`-AllowDirty` overrides), runs `go vet`, the unit tests and the GPU tests (`-SkipGpu` skips those), rebuilds `dawnrt.exe` when `rt.cc` is newer, and writes `dist\gptrun-<release>.zip`: `gptrun.exe` stamped with the release, `dawnrt\` with its DLLs, and `manifest.txt` with the commits, a hash of each dependency tree, and every file's SHA-256. `EmbedDB` is not a git repository, so the manifest records its tree hash only. A release finds `dawnrt` beside `gptrun.exe`.

### Accuracy

```
gptrun eval --config C:\path\to\model.bnz.yaml
```

`eval/answers.jsonl` holds questions with the route the think pass should pick and the words the answer must and must not contain. Each is answered the way `gptrun run` answers it, routed as the chat screen routes it. Command questions only check the route unless `-exec` runs them in the guest. The report goes to `eval/results/`; `-min 0.8` fails below 80% answer accuracy.

## Paths

Defaults sit under the home directory. An environment variable replaces the whole path.

| Variable | Default | Used for |
| --- | --- | --- |
| `MODELRUN_APOLLO` | `~/WASMApollo` | WGSL tree. Also sets `dawn.exe` when `MODELRUN_DAWN` is empty |
| `MODELRUN_DAWN` | `~/WASMApollo/dawn/dawn.exe` | LUT runner and the clock |
| `MODELRUN_ENCRYPTID` | `~/EncryptidSystems/stage-ai` | Kernel directory `kernels/` |
| `MODELRUN_DAWNHOST` | `~/ModelRun/dawnhost/dawnhost.exe` | Resident Dawn host |
| `MODELRUN_DAWNRT` | `dawnrt\` beside `gptrun.exe`, else `~/ModelRun/dawnrt/dawnrt.exe` | Device process for a Hodge dispatch |
| `MODELRUN_SKILLS` | `~/ModelRun/skills` | Skill files |
| `MODELRUN_SEARCH` | `https://ai.encryptid.systems` | Fact lookup |
| `MODELRUN_VM` | `http://127.0.0.1:8870/mcp` | TunnelTug guest |
| `MODELRUN_SYSTEM` | built-in device prompt | System prompt. A path to a file, or the prompt text |

Kernels the runtime loads from `kernels/` include `hodgelinear.wgsl`, `dense.wgsl`, `dot4_soft.wgsl`, `dot4_native.wgsl`, `embed.wgsl`, `norm.wgsl`, `rope.wgsl`, and `attention.wgsl`. ModelRun's own kernels are compiled into the program from `internal/hodge/kernels`. Speed lists them.

## Checkpoint file

`gptrun run --config` reads the first entry. Paths are relative to the config file. Unknown fields are an error.

```yaml
models:
  - model: model.safetensors
    config: config.json
    tokenizer: tokenizer.json
    tokenizer_config: tokenizer_config.json
    generation_config: generation_config.json
```

`model` is safetensors or BNZ4. `generation_config` and `special_tokens_map` may be omitted. `config.json` supplies `hidden_size`, `num_hidden_layers`, and `vocab_size`.

## Commands

```
gptrun list
gptrun show <model>
gptrun ps
gptrun run [-n N] [-temp T] [-top-p P] [-system TEXT] --config FILE [text]
gptrun run [-n N] [-temp T] [-top-p P] [-system TEXT] <model> [text]
gptrun serve [-addr 127.0.0.1:11434]
```

`list` and `ls` print the clock, then every WGSL kernel whose name contains lut, fabric, ternary, hodge, embed, dot4, or kuramoto. `show` prints one catalog row. `ps` prints models loaded in this process.

`run` without a prompt, on a terminal, opens the chat screen. A pipe with an empty prompt is read as the prompt.

Omit `-n` and the answer runs until the model stops. `-n N` caps the length. Omit `-temp` and `-top-p` and the checkpoint sampler is used. `-temp 0` is greedy.

`gptrun run dawn/clock` runs the Dawn clock. `gptrun run apollo/<kernel>` loads that WGSL file in `dawn.exe`. Those catalog runs are one shot. They are not a chat checkpoint.

## Chat screen

The screen is the checkpoint path. Enter sends. Shift+Enter or Alt+Enter starts a new line. The wheel scrolls the transcript. Esc twice quits. Ctrl+C stops the current reply, and quits when nothing is running. Alt+C copies the latest reply. A finished reply also has copy, up, and down on its own line. Up keeps that reply. Down asks RHFL for the lookup when the reply stood, and puts the model's own words back when a lookup had replaced them.

| Input | Effect |
| --- | --- |
| `/tools` | List exec, search, and gather |
| `/skills` | List skill files |
| `/skill name` | Turn that skill on |
| `/skill` | Turn the skill off |
| `/system` | Show the system prompt |
| `/system text` | Use that prompt on the next question |
| `/system path` | Read the prompt from that file |
| `/system default` | Clear the session prompt |
| `/run …` or `/exec …` | Run that command in the guest |

A skill is a `.md` file in the skills directory. The first heading is dropped. The first remaining line is the summary.

The system prompt is the built-in device text unless `MODELRUN_SYSTEM` or `/system` replaces it. `-system` does the same for one run. A path that names a file is read, up to 32KiB. Any other value is the prompt itself. Changing it drops the chat cache. The next question is prefilled with the new text. Fact, recipe, and command passes keep their own instructions.

The line under a finished reply shows that answer's decode rate, such as `42 tok/s`.

A conversation longer than the context window keeps going: when the next turn would not fit, the chat is rebuilt from the system prompt and the latest turns, about a quarter of the window, and the status line says so. One message too large for the window is refused and the conversation is kept.

When the GPU process stops (a crash, a driver reset, sleep, or no reply for three minutes) the next turn loads the model again and rebuilds the chat from its text. A reply cut off in the middle is not restarted on screen; the error says to send it again.

A typo is read as the nearest word the checkpoint already knows, when that word is clearly nearer than any other. The line you typed stays in the transcript. The model hears the recovered wording, and the screen shows it under spell. Words from the Ailza list are left as written.

The checkpoint answers in the conversation, and that reply is streamed as it is written. RHFL reads the finished reply. A conversational reply stands. A bare fact label, an empty checkable answer, or a short decline of a live or checkable fact is corrected from a lookup that reads the subject's page. A sentence that names something the notes do not, a name or a number, is dropped; when nothing is left, the notes are shown. A tool call in the reply runs, including when the model also wrote a sentence. search looks something up, gather reads one public page, and exec runs in the guest. The model then answers in plain words. A recipe stays with the model. The line under the reply names that RHFL choice. copy puts the reply on the clipboard. up agrees with it. down disagrees: a reply that stood is looked up, and a reply that was replaced returns to the model's own words.

## Speed

Every speedup below gives the bits the plain path gives: the same logits, the same cache, the same tokens. The tests in `internal/hodge` check that on a real checkpoint with `MODELRUN_GOLDEN` set.

| Part | What runs |
| --- | --- |
| Device process | One submit per token: queue writes, every layer, the head, and the readback copy in one command buffer. Waits block on the device fence. Robustness checks and workgroup zero-init are off |
| Linears | `hlx.wgsl` runs q, k and v as one dispatch, and gate and up as one. Every limb of a weight sums in that dispatch. o and down add into the residual stream there. The hardware dot4 is used when the device has it |
| Prefill | Up to 32 prompt tokens per submit. `hlxb.wgsl`, `normb.wgsl`, `ropeb.wgsl`, `attngb.wgsl`, and `embedb.wgsl` read each weight once per 8 tokens |
| Head | `lmscreen.wgsl` reads an int8 copy of the embedding (`lmquant.wgsl`, at load). Its error bound picks the rows the sampler can reach, and `lmgather.wgsl` computes those exactly. A flat distribution at a high temperature runs the full `lmhead.wgsl` |
| Attention | `attng.wgsl` reads each cached row once per grouped-query group, not once per head |
| Sampler | Sorts only the tokens min-p keeps and the end tokens, not the whole window |
| Cache | The device cache records whose tokens it holds. A prompt that starts with cached tokens runs only the rest. A chat whose rows another chat wrote over runs them again |

These switch a part off:

| Variable | Effect |
| --- | --- |
| `MODELRUN_HEAD=exact` | The full head on every token |
| `MODELRUN_BATCH=0` | Prefill one token per submit |
| `MODELRUN_FUSE=0` | One `hodgelinear.wgsl` dispatch per limb, and a separate residual add |
| `MODELRUN_DOT4=soft` | `dot4_soft.wgsl` even when the device has the instruction |
| `MODELRUN_ATTN=head` | One attention workgroup per head |
| `MODELRUN_DAWN_SAFE=1` | Dawn's robustness checks and workgroup zero-init on |
| `MODELRUN_DAWN_TIME=1` | `dawnrt` prints each submit's encode, submit and wait time |

`TestProfile` times the parts of one token on a checkpoint:

```
set MODELRUN_PROF=C:\models\hodge-5\hodge-5.bnz.yaml
go test -run TestProfile -v ./internal/hodge
```

## HTTP

`gptrun serve` listens on `127.0.0.1:11434`. The routes are the Ollama set. `stream` is off unless the request sets it to true. A stream is newline-delimited JSON. The `model` field is a catalog name from `gptrun list`.

| Method | Path |
| --- | --- |
| GET | `/api/version` |
| GET | `/api/tags` |
| GET | `/api/ps` |
| POST | `/api/show` |
| POST | `/api/chat` |
| POST | `/api/generate` |

`/api/chat` reads `messages`. `/api/generate` reads `prompt` and does not apply the chat template. Sampler fields live under `options`: `temperature`, `top_p`, and `num_predict`.

```
curl -s http://127.0.0.1:11434/api/tags
curl -s http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" -d "{\"model\":\"dawn/clock\",\"stream\":false}"
```

## Watermark

When `model.cochain.safetensors` sits beside the tokenizer, each checkpoint reply is scored against that cochain. `gptrun run` prints the reading on stderr:

```
watermark  clear  words 40  alignment 0.120  null 0.080  cell 0.0100  cell_null 0.0040
```

The first word is `marked`, `clear`, or `short` (under 24 words). The HTTP object carries the same reading as `watermark` when the model has a cochain.

## Corpus tools

These two programs read a cochain. They are not the chat runtime.

`cochainkey` prints the shared sign key of a folder of `.txt` files. Its config is the checkpoint file above, and it also accepts `tensor` when the safetensors file has more than one matrix. `--tensor` overrides that field.

```
cochainkey --config FILE CORPUS
```

`cochainimg` draws one chain per text, then the shared centroid. The model directory contains `tokenizer.json`, `tokenizer_config.json`, and `model.cochain.safetensors`.

```
cochainimg -model MODEL -dir DIR -out cochains.png
```

## Tree

```
cmd/gptrun     CLI and chat screen
cmd/cochainkey   corpus sign key
cmd/cochainimg   cochain picture
internal/hodge   checkpoint runtime
internal/api     HTTP routes
dawnhost/        resident Dawn host
dawnrt/          Hodge device process
```
