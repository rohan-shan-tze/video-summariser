# Writeup

## What the system does

A fully local, offline AI desktop app. The user picks a short `.mp4` file and
chats with it in natural language. The system can:

1. **Transcribe** - speech-to-text via faster-whisper
2. **Detect objects** - frame sampling + YOLOv8n via OpenVINO IR
3. **Detect graphs / extract text** - EasyOCR with a keyword heuristic
4. **Summarize** - two backends: Llama 3.2 1B Instruct (local LLM, default) or
   LexRank extractive sentence ranking (request with "extractive summary")
5. **Generate a PDF report** - fpdf2
6. **Generate a PowerPoint** - python-pptx

All five assignment spec queries work end-to-end. Chat history persists across
restarts via SQLite. The UI is a Tauri desktop app (Rust core + React webview).

---

## What works

- All five spec queries pass end-to-end through the full stack.
- Intent classification with confidence scoring and human-in-the-loop clarification.
- Multi-step chains (e.g. "summarize and generate a PDF" triggers transcribe -> summarize -> make_pdf automatically).
- Session state caching: if the user already asked for a transcript, a follow-up "summarize" reuses it without re-transcribing.
- Cache invalidation when the user switches to a different video.
- SQLite persistence: every turn (user + assistant) is saved and survives backend restarts.
- Session history browser: a "[ SESSIONS ]" button in the header lists all past sessions. Clicking one rehydrates the full message history from SQLite via gRPC (`GetHistory` RPC).
- Session video restore: switching to a past session also restores the video path that was active in that session (stored in the `sessions` table). If the file no longer exists the user is simply shown no video selected.
- Clickable artifact links: generated PDF and PPTX notices in the chat are clickable and open the file with the OS default application via `tauri-plugin-opener`.
- Real MCP architecture: each capability is a genuine subprocess communicating over stdio JSON-RPC. The orchestrator never calls inference directly.
- gRPC bound to 127.0.0.1 only, no network exposure.
- LLM summarization: Llama 3.2 1B Instruct Q4_K_M GGUF runs locally via `llama-cpp-python` with
  `n_gpu_layers=0` (CPU-only). The model produces natural, synthesised summaries rather than
  extracting raw sentences. The extractive LexRank backend is preserved and selectable by the user.
- CPU-first inference: all defaults are `device="cpu"` (Whisper), `device_name="AUTO"` (OpenVINO),
  `gpu=False` (EasyOCR), `n_gpu_layers=0` (Llama). No CUDA wheels.

---

## What is stubbed and how to finish it

**Open-ended vision queries**
The current system handles six fixed intents. A query like "does this video have
a whiteboard?" falls through to the ambiguous handler. To finish: after the
rule-based classifier gives up, pass the query text plus the session's cached
`text_regions` and `objects` to the already-present Llama 3.2 1B model for
open-ended Q&A against the extracted vision data. The local LLM infrastructure
is already in place - this is purely an orchestrator routing addition.

---

## Tradeoffs made under time pressure

**Keyword-based intent classification**
A learned classifier (TF-IDF + logistic regression, or a small BERT) would
handle paraphrases and edge cases better. The keyword approach was chosen because
it is transparent, requires no training data, and every routing decision can be
explained line by line in an interview. The `_classify()` function returns a
typed `(intent, confidence)` tuple, so a learned classifier could replace the body
without touching anything else.

**No streaming transcription to the UI**
faster-whisper returns a segment generator. The current implementation collects
all segments before replying. For long videos, this means a silent wait. The
fix is to use gRPC server-side streaming (`server_streaming` RPC) and push each
segment to the UI as it arrives.

**LLM summarization is slower than extractive on first call**
The Llama 3.2 1B model loads into the MCP subprocess on the first summarize
request (15-30 seconds on CPU). Subsequent calls in the same session are faster
since the model stays resident. The extractive LexRank backend is instant and
available as an explicit user option ("extractive summary"). The tradeoff: LLM
produces more natural, synthesised output but requires the model to be downloaded
(~800 MB) and has higher latency. LexRank is deterministic, zero-download, and
returns in under a second, but can only select existing sentences verbatim.

---

## Challenges encountered

**protoc not available in the build environment**
Rust's tonic-build requires the Protocol Buffer compiler (`protoc`). Fixed by
adding `protoc-bin-vendored` to Cargo build-dependencies, which ships the binary
and sets the `PROTOC` env var in `build.rs` automatically. No system install required.

**Serde snake_case vs camelCase mismatch**
Rust struct fields (`needs_clarification`, `artifact_path`) were serialized as
snake_case but React read them as camelCase (`resp.needsClarification`). This
silently broke clarification option rendering. Fixed with
`#[serde(rename_all = "camelCase")]` on the `ChatResponse` struct.

**OpenVINO model sourcing**
The original plan used `omz_downloader` from `openvino-dev` to fetch MobileNet-SSD.
`openvino-dev` does not install on Python 3.13 and the Open Model Zoo storage
server no longer serves direct binary downloads. Switched to YOLOv8n: export to
ONNX via ultralytics, convert to OpenVINO IR with `ovc` (bundled in the main
`openvino` package). YOLOv8n is a better model (COCO 80 classes, higher accuracy)
at a comparable size.

**NLTK data discovery raising OSError**
`nltk.data.find()` raises `OSError` (not just `LookupError`) when the data path
exists but the resource layout inside it is wrong. The NLTK bootstrap now catches
both exceptions.

**EasyOCR model load time**
First-run EasyOCR loads ~100 MB of weights. The gRPC test client originally used a
120-second timeout which expired before the model finished loading. Bumped to 300s
for the vision_text_graph test; in production the model stays loaded in the
subprocess after first use.

---

## What more time would add

1. **Streaming UI** - gRPC server-streaming so the transcript appears word by word as Whisper produces it.
2. **LLM-backed Q&A** - extend the already-present Llama 3.2 1B model to handle open-ended
   questions about the video using the cached transcript, objects, and text as context.
   The model and MCP infrastructure are already in place; this is an orchestrator routing addition.
3. **Confidence tuning UI** - expose `_CLARIFICATION_THRESHOLD` as a setting so the user can make the classifier more or less aggressive.
4. **C# launcher** - one-click launch of the packaged Python backend from a small C# executable (PyInstaller + C# process manager). See Phase 11 in the build plan.
5. **Batch video analysis** - accept a folder of `.mp4` files and produce a combined report.
