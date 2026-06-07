"""
Transcription MCP Server.

Exposes one tool:
  transcribe(video_path) -> {text: str, segments: [{start, end, text}]}

Internally uses faster-whisper (base model, int8, CPU) to perform speech-to-text.
faster-whisper calls system ffmpeg under the hood to decode audio from the mp4 —
no explicit audio extraction step needed in Python.

Run as a subprocess; communicates via stdio JSON-RPC (see mcp_common/server.py).
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so backend.mcp_common resolves
# regardless of where the process is launched from.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.mcp_common.server import MCPServer

# Model is loaded once at startup and reused for every transcribe() call.
# Loading inside the class init (not at module level) so import errors for
# missing packages surface as MCP error responses rather than a crashed process.
_model = None


def _get_model():
    """Lazy-load the Whisper model on first transcription call."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        models_dir = Path(__file__).parent.parent.parent / "models" / "whisper"
        # device="cpu" and compute_type="int8" are the committed defaults.
        # Never change these to "cuda" in committed code. 
        # I will assume reviewer machine is CPU-only.
        _model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            download_root=str(models_dir),
        )
    return _model


def transcribe(video_path: str) -> dict:
    """
    Transcribe the audio track of an mp4 file.

    Returns:
      text.     full transcript as a single string
      segments. list of {start, end, text} dicts (timestamps in seconds)
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = _get_model()

    # transcribe() returns a generator of Segment objects + metadata.
    # beam_size=5 is the faster-whisper default; vad_filter suppresses
    # silent gaps so the transcript is cleaner on short clips.
    segments_iter, _info = model.transcribe(
        str(path),
        beam_size=5,
        vad_filter=True,
    )

    segments = []
    full_text_parts = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end":   round(seg.end,   2),
            "text":  seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    return {
        "text":     " ".join(full_text_parts),
        "segments": segments,
    }


if __name__ == "__main__":
    server = MCPServer("transcription_server")
    server.register_tool("transcribe", transcribe)
    server.run()
