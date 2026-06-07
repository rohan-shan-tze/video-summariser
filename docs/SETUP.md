# Setup Guide

Everything needed to go from `git clone` to a running app. Follow in order.
Tested on Windows 11. The backend is pure Python and runs on any OS; the
Tauri frontend targets Windows/macOS/Linux.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.10 - 3.12 | 3.13 is NOT supported (openvino incompatibility; no llama-cpp-python wheel) |
| Rust + Cargo | 1.75+ | Install via [rustup.rs](https://rustup.rs/) |
| Node.js | 18+ | For the Tauri frontend dev toolchain |
| npm | 9+ | Comes with Node.js |

> **Windows note:** after installing Rust via rustup, restart your terminal
> (or VS Code) so the new `%USERPROFILE%\.cargo\bin` PATH entry is picked up.

---

## 1. Clone the repo

```bash
git clone <repo-url>
cd video-summariser
```

---

## 2. Python virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

The first command installs gRPC, faster-whisper, OpenVINO, EasyOCR, sumy,
fpdf2, python-pptx, and their dependencies. No CUDA wheels are included - the
app runs entirely on CPU.

The second command installs `llama-cpp-python` from the maintainer's pre-built
CPU wheel index. It must be run separately because the source build requires
a C++ compiler on Windows and often fails due to setuptools version mismatches.
The `--extra-index-url` flag pulls a pre-compiled binary so no compilation is needed.

---

## 4. Download models (one-time, ~1.1 GB)

```bash
python download_models.py
```

This downloads and sets up four models into the `models/` directory:

| Model | Size | Purpose |
|---|---|---|
| faster-whisper `base` int8 | ~150 MB | Speech-to-text transcription |
| YOLOv8n -> OpenVINO IR | ~18 MB (ONNX + IR) | Object detection |
| EasyOCR English | ~100 MB | On-screen text extraction |
| Llama 3.2 1B Instruct Q4_K_M | ~800 MB | LLM-based summarization (CPU-only) |

**First run downloads, then fully offline.** After this step the app makes no
network calls at runtime.

> **OpenVINO model note:** `download_models.py` exports YOLOv8n to ONNX and
> converts it to OpenVINO IR. This requires `ultralytics` which is NOT in
> `requirements.txt` (it is only needed once):
>
> ```bash
> pip install ultralytics==8.4.60
> python download_models.py
> ```
>
> If the model already exists at `models/openvino/public/mobilenet-ssd/FP32/mobilenet-ssd.xml`
> the download step is skipped automatically.

---

## 5. Start the Python backend

Open a terminal in the repo root (with the venv activated):

```bash
python backend/grpc_server.py
```

Expected output:
```
[grpc_server] Listening on 127.0.0.1:50051
```

Leave this terminal running. The backend binds to localhost only - it is never
accessible from outside the machine.

> **First query note:** EasyOCR loads its model on the first vision request.
> This can take 30-60 seconds. Subsequent calls are fast (model stays loaded
> in the subprocess).

---

## 6. Run the frontend

Open a second terminal in `frontend/`:

```bash
cd frontend
npm install
npm run tauri dev
```

The Tauri desktop window will open. During development, Vite serves the React
app on `http://localhost:1420` and Rust rebuilds automatically on file changes.

> **First Rust build:** `cargo build` compiles the Rust gRPC client and
> generates protobuf stubs. This takes 2-5 minutes on a cold cache.
> Incremental rebuilds are fast.

---

## 7. Using the app

1. Click **[ PICK VIDEO ]** and select a `.mp4` file (tested with ~1-minute clips).
2. Type a message and press **Enter** (or click **SEND**).
3. To resume a prior conversation, click **[ SESSIONS ]** and select a past session.

Example queries:
- `Transcribe the video`
- `What objects appear in the video?`
- `Are there any graphs or charts?`
- `Summarize the video` (uses Llama 3.2 1B LLM by default)
- `Extractive summary` (uses LexRank - instant, no model load)
- `Create a PowerPoint with the key points`
- `Summarize our discussion so far and generate a PDF`

> **First LLM summarize call:** Llama 3.2 1B loads into the summarization
> subprocess on demand (~15-30 seconds on CPU). Subsequent calls in the same
> session are faster. Use "extractive summary" if you want an instant result.

Generated PDF and PPTX files are written to the `outputs/` folder at the repo
root. The chat UI shows a **FILE READY** notice when a file is generated - click
it to open the file with your default application.

---

## 8. Sample video and queries

A sample `.mp4`s and five spec queries are in `samples/`:

```
samples/
  psychology-lecture.mp4   - MIT OCW Introduction to Psychology excerpt
  law-lecture.mp4          - Emerson Stafford Equal Protection Lecture excerpt
  sample_queries.md        - all five spec queries with expected behaviour
```

---

## Troubleshooting

**`Could not connect to backend`** in the UI
- The Python gRPC server is not running. Start it with `python backend/grpc_server.py`.

**`OpenVINO model not found`**
- Run `python download_models.py` (see step 4). Ensure `ultralytics` is installed first.

**EasyOCR hangs on first vision query**
- Normal - EasyOCR loads ~100 MB of model weights on first use. Wait up to 60 seconds.

**`protoc not found` during Rust build**
- The build uses `protoc-bin-vendored` so no system protoc install is needed.
  If this error appears, ensure you are building from `frontend/src-tauri/` and that
  `Cargo.toml` includes `protoc-bin-vendored = "3"` in `[build-dependencies]`.

**Port 50051 already in use**
- A previous backend instance is still running. On Windows: `netstat -ano | findstr 50051`
  to find the PID, then `taskkill /PID <pid> /F`.

**Python 3.13 compatibility**
- Use Python 3.10-3.12. `openvino` does not support 3.13 at time of writing.

---

## Folder layout reference

```
/backend          - Python gRPC server + MCP servers + orchestrator + storage
/frontend         - Tauri app (Rust core + React webview)
/models           - downloaded models (gitignored)
/outputs          - generated PDFs and PPTX files (gitignored)
/samples          - sample videos
/sample_outputs   - example outputs 
/ui-images        - snapshots of the UI
/docs             - this file + ARCHITECTURE.md + WRITEUP.md + BUILD_LOG.md
download_models.py
requirements.txt
```
