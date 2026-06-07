# Video Summariser

A fully local, offline AI desktop app for analysing short `.mp4` videos via a chat interface.

Pick a video, ask questions in natural language — transcribe it, detect objects, find graphs,
summarize, generate a PDF or PowerPoint. Nothing leaves the machine.

---

## Features

- **Transcription** - speech-to-text via faster-whisper (Whisper base, CPU int8)
- **Object detection** - YOLOv8n via OpenVINO IR, 80 COCO classes
- **Text / graph detection** - EasyOCR frame sampling with graph keyword heuristic
- **Extractive summarization** - LexRank sentence ranking (sumy + NLTK), no model download
- **PDF generation** - fpdf2
- **PowerPoint generation** - python-pptx
- **Chat history persistence** - SQLite, survives backend restarts
- **Human-in-the-loop clarification** - low-confidence queries return option buttons

## Architecture

```
React (webview) -> Rust (tonic gRPC) -> Python gRPC server -> Orchestrator
                                                                    |
                          Transcription MCP  Vision MCP  Summarization MCP  Generation MCP
                          (faster-whisper)  (OpenVINO)   (LexRank/sumy)    (fpdf2/pptx)
                                                               SQLite
```

Each capability is a real MCP server: a subprocess communicating via stdio JSON-RPC.
The orchestrator routes, sequences, and manages state. It never runs inference directly.

## Quick start

See [docs/SETUP.md](docs/SETUP.md) for the full setup guide.

```bash
# 1. Install Python deps
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Download models (one-time, ~300 MB)
pip install ultralytics==8.4.60   # only needed for model conversion
python download_models.py

# 3. Start the backend
python backend/grpc_server.py

# 4. Start the desktop app (new terminal)
cd frontend && npm install && npm run tauri dev
```

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): component diagram and design decisions
- [docs/SETUP.md](docs/SETUP.md): full setup guide from clone to running app
- [docs/WRITEUP.md](docs/WRITEUP.md): what works, stubs, tradeoffs, and challenges
- [docs/BUILD_LOG.md](docs/BUILD_LOG.md): per-phase build log

## Sample queries

| Query | Intent |
|---|---|
| `Transcribe the video` | Transcription |
| `What objects appear in the video?` | Object detection |
| `Are there any graphs or charts?` | Graph / text detection |
| `Summarize the video` | Extractive summarization |
| `Create a PowerPoint with the key points` | PPTX generation |
| `Summarize our discussion so far and generate a PDF` | Summarization + PDF chain |

See [samples/sample_queries.md](samples/sample_queries.md) for the full spec query set.

## Requirements

- Python 3.10 - 3.12 (3.13 not supported)
- Rust 1.75+ (via [rustup.rs](https://rustup.rs/))
- Node.js 18+
- CPU-only machine supported; GPU optional (OpenVINO AUTO device policy)
