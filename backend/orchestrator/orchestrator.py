"""
Orchestrator — intent router and conversation state manager.

Phase 1 scope: handles exactly ONE intent (transcribe). The full intent
classifier, confidence scoring, clarification logic, and multi-step chains
are built in Phase 5. This version is intentionally minimal so we can prove
the end-to-end chain works before adding breadth.

How it works (Phase 1):
  1. Receive a ChatRequest (session_id, text, video_path).
  2. Detect "transcribe" intent by simple keyword match.
  3. If matched: call the Transcription MCP server via MCPClient, return transcript.
  4. Otherwise: return a placeholder "not yet implemented" reply.

Conversation state is a dict keyed by session_id. Each session stores the
transcript (and later: detected objects, prior turns, etc.) so follow-up
queries can reference earlier results without re-running inference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.mcp_common.client import MCPClient


# Per-session state

# Keyed by session_id (string). Each value is a dict with whatever the session
# has accumulated so far. Phase 1 only stores the transcript.
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    """Return (creating if absent) the state dict for this session."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "transcript": None,   # str once transcribed
            "segments":   None,   # list[dict] once transcribed
            "video_path": None,   # last video path seen
        }
    return _sessions[session_id]


# MCP client: one instance, lazily spawned, reused across calls

# The client is module-level so it persists across requests and the server
# subprocess is only spawned once (not once per query).
_transcription_client: MCPClient | None = None


def _get_transcription_client() -> MCPClient:
    global _transcription_client
    if _transcription_client is None:
        server_script = Path(__file__).parent.parent / "mcp_servers" / "transcription_server.py"
        _transcription_client = MCPClient([sys.executable, str(server_script)])
    return _transcription_client



# Intent detection (Phase 1: keyword-only, single intent)

# In Phase 5 this becomes a full classifier with confidence scores and
# ambiguity handling. Here it's intentionally transparent and simple so
# the vertical slice is easy to reason about.

_TRANSCRIBE_KEYWORDS = {"transcribe", "transcript", "speech", "audio", "words", "said", "say"}


def _is_transcribe_intent(text: str) -> bool:
    """Return True if the query looks like a transcription request."""
    lowered = text.lower()
    return any(kw in lowered for kw in _TRANSCRIBE_KEYWORDS)


# Public entry point called by the gRPC server

def handle(session_id: str, text: str, video_path: str) -> dict:
    """
    Process one chat turn and return a response dict matching ChatResponse fields:
      reply               - text to show in the chat bubble
      needs_clarification - always False in Phase 1
      options             - always [] in Phase 1
      artifact_path       - always "" in Phase 1

    Raises nothing. All exceptions are caught and returned as error replies
    so the gRPC server always has something to send back.
    """
    state = _get_session(session_id)

    # Update stored video path if one was provided this turn.
    if video_path:
        state["video_path"] = video_path

    effective_video = state["video_path"]

    try:
        if _is_transcribe_intent(text):
            return _handle_transcribe(state, effective_video)
        else:
            # Phase 1 stub: any non-transcribe query gets a polite placeholder.
            return _reply("I can only transcribe videos right now. More capabilities coming soon.")

    except Exception as exc:
        return _reply(f"Error: {exc}")


def _handle_transcribe(state: dict, video_path: str | None) -> dict:
    """Call the Transcription MCP server and cache the result in session state."""
    if not video_path:
        return _reply("Please select a video file before asking me to transcribe.")

    # If we already have a transcript for this session, return it from cache
    # rather than re-running inference. Re-transcription can be forced by
    # selecting a new video (which updates state["video_path"]).
    if state["transcript"] is not None:
        return _reply(f"Transcript (cached):\n\n{state['transcript']}")

    client = _get_transcription_client()
    # This call blocks until the MCP server responds.
    # MCPClient.call() raises RuntimeError on server-side errors.
    result = client.call("transcribe", video_path=video_path)

    # Cache in session state for follow-up queries (e.g. "summarize this").
    state["transcript"] = result["text"]
    state["segments"]   = result["segments"]

    return _reply(f"Transcript:\n\n{result['text']}")


def _reply(text: str) -> dict:
    """Construct a minimal ChatResponse-shaped dict."""
    return {
        "reply":               text,
        "needs_clarification": False,
        "options":             [],
        "artifact_path":       "",
    }


def shutdown():
    """Clean up MCP subprocesses — call before the gRPC server exits."""
    global _transcription_client
    if _transcription_client is not None:
        _transcription_client.shutdown()
        _transcription_client = None
