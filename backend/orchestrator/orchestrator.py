"""
Orchestrator — intent router, state manager, and clarification handler.

This is the agentic core of the system. It is the ONLY component that knows
about the other MCP servers and how to sequence calls between them.

Responsibilities:
  1. Intent classification   map user text -> intent + confidence score
  2. Routing/dispatch        map intent -> MCP tool call(s), including multi-step chains
  3. Conversation state      per session_id, cache inference results and prior turns
  4. Human-in-the-loop       if confidence is low or input is missing, ask instead of guess

What the orchestrator does NOT do:
  - It never runs inference itself (no Whisper, no OpenVINO, no LexRank here).
  - It never generates document content (that is the Generation MCP server's job).
  - It never summarizes text (that is the Summarization MCP server's job).

Architecture reminder:
  gRPC server -> orchestrator.handle() -> MCPClient.call() -> MCP server subprocess
  All MCP calls cross a subprocess boundary via stdio JSON-RPC.

INTENT CLASSIFICATION  


Intents and what they map to:

  TRANSCRIBE        -> transcription_server.transcribe()
  VISION_OBJECTS    -> vision_server.detect_objects()
  VISION_TEXT_GRAPH -> vision_server.extract_text()
  SUMMARIZE         -> [transcribe if needed] -> summarization_server.summarize()
  GENERATE_PDF      -> [transcribe+summarize if needed] -> generation_server.make_pdf()
  GENERATE_PPTX     -> [transcribe+summarize if needed] -> generation_server.make_pptx()
  AMBIGUOUS         -> return needs_clarification=True with options

Classification is keyword/rule-based. This is a deliberate choice:
  - Transparent: every routing decision can be explained line by line.
  - Defensible in an interview: no black box.
  - Replaceable: the classify() function returns a (intent, confidence) tuple.
    A learned classifier (e.g. tiny TF-IDF + logistic regression) could replace
    the body of classify() without changing anything else in this file.

Confidence scoring:
  Each intent has a set of "strong" keywords (unambiguous) and "weak" keywords
  (suggestive but could match other intents). The score is a float in [0, 1]:
    - 1.0  strong keyword matched
    - 0.6  only weak keywords matched
    - 0.0  no keywords matched -> AMBIGUOUS

  If confidence < _CLARIFICATION_THRESHOLD we do not guess — we return a
  clarification prompt with options. This satisfies the spec requirement for
  human-in-the-loop when intent is uncertain.


CONVERSATION STATE


_sessions[session_id] holds everything accumulated in one chat session:

  video_path       str | None    most recently provided video file path
  transcript       str | None    full transcript text (cached after first transcription)
  segments         list | None   [{start, end, text}] timed segments
  objects          list | None   [{label, confidence, frame_ts}] from vision
  text_regions     list | None   [str] OCR text from vision
  likely_has_graph bool | None   graph heuristic from vision
  summary          dict | None   {key_points, summary} from summarizer
  artifact_path    str  | None   path of last generated PDF or PPTX
  pending_intent   str  | None   intent we asked the user to clarify
  turns            list          [{role, text}] full conversation history

The cached results mean: if a user asks "summarize the video" and already asked
"transcribe the video" earlier in the same session, we skip re-transcription and
go straight to summarization. This is important for responsiveness.

CLARIFICATION FLOW

When we ask for clarification:
  1. We set needs_clarification=True and return option strings in the response.
  2. We save the ambiguous intent in state["pending_intent"].
  3. On the next turn, _resolve_clarification() checks if the user's text
     matches one of the pending options. If it does, we route to that intent.
     If it doesn't, we classify the new message fresh (user may have rephrased).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.mcp_common.client import MCPClient


# Constants

# If the top-scoring intent has confidence below this threshold, we ask
# for clarification instead of guessing.
_CLARIFICATION_THRESHOLD = 0.7


# Intent definitions

# Each entry: (intent_name, strong_keywords, weak_keywords)
# strong match -> confidence 1.0 -> routes directly (above _CLARIFICATION_THRESHOLD)
# weak match   -> confidence 0.6 -> always ambiguous, but ranks the clarification options
#   so "I want to see the things in the video" surfaces "Detect objects" first
#   rather than showing the full generic capability menu.
_INTENT_RULES = [
    (
        "transcribe",
        {"transcribe", "transcript", "transcription"},
        {"speech", "audio", "words", "said", "say", "spoken", "hear", "listen"},
    ),
    (
        "vision_objects",
        {"objects", "object", "detect", "detection", "appear", "appears", "shown", "show"},
        {"see", "visible", "scene", "frame", "person", "people", "thing", "things", "what is"},
    ),
    (
        "vision_text_graph",
        {"graph", "chart", "graphs", "charts", "plot", "plots", "diagram"},
        {"text", "ocr", "written", "slide", "slides", "figure", "table", "visual"},
    ),
    (
        "summarize",
        {"summarize", "summarise", "summary", "summarization", "key points", "key takeaways", "main points"},
        {"overview", "recap", "brief", "short", "condense", "highlights"},
    ),
    (
        "generate_pdf",
        {"pdf", "PDF"},
        {"report", "document", "file", "generate", "create", "make", "export"},
    ),
    (
        "generate_pptx",
        {"pptx", "powerpoint", "presentation", "slides", "slide deck"},
        {"deck", "keynote"},
    ),
]


# Intent classification


def _classify(text: str) -> list[tuple[str, float]]:
    """
    Score every intent against the user's message.

    Returns a list of (intent, confidence) sorted by confidence descending.
    Confidence is in {0.0, 0.6, 1.0}, simple discrete scale.
      1.0  strong keyword matched -> routes directly
      0.6  weak keyword matched  -> still ambiguous (below threshold), but the
           score is used to rank clarification options so the most likely intent
           appears first rather than showing a generic full-menu prompt.
      0.0  no match              -> not included in scores list

    Why discrete? Continuous scoring would require training data. Discrete
    levels are transparent and easy to explain: strong keywords are those that
    almost never appear outside their intent; weak keywords are suggestive but
    too ambiguous to route on alone.
    """
    lowered = text.lower()
    words = set(lowered.split())  # rough word set for single-word keyword matching

    scores = []
    for intent, strong, weak in _INTENT_RULES:
        # Check strong keywords first — any single strong match is sufficient.
        # We check both word boundaries (words set) and substring (for phrases
        # like "key points" which won't appear as a single element in words).
        strong_hit = any(kw in words or kw in lowered for kw in strong)
        weak_hit   = any(kw in words or kw in lowered for kw in weak)

        if strong_hit:
            scores.append((intent, 1.0))
        elif weak_hit:
            scores.append((intent, 0.6))
        # score 0.0 -> don't include; treated as AMBIGUOUS below

    # Sort by confidence descending so the best match is first.
    scores.sort(key=lambda x: -x[1])
    return scores


def _pick_intent(scores: list[tuple[str, float]]) -> tuple[str, float]:
    """
    Given the ranked scores from _classify(), decide which intent to act on.

    Rules:
      - No matches at all         -> ambiguous, confidence 0.0
      - Top score < threshold     -> ambiguous (low confidence)
      - Two intents tied at 1.0   -> ambiguous (can't distinguish)
      - Otherwise                 -> top intent wins
    """
    if not scores:
        return "ambiguous", 0.0

    top_intent, top_conf = scores[0]

    # Two or more intents with equal top confidence and both strong -> ask.
    if len(scores) > 1 and scores[1][1] == top_conf == 1.0:
        return "ambiguous", top_conf

    if top_conf < _CLARIFICATION_THRESHOLD:
        return "ambiguous", top_conf

    return top_intent, top_conf


# Session state

_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    """Return (creating if absent) the state dict for this session."""
    if session_id not in _sessions:
        _sessions[session_id] = {
            "video_path":       None,
            "transcript":       None,
            "segments":         None,
            "objects":          None,
            "text_regions":     None,
            "likely_has_graph": None,
            "summary":          None,
            "artifact_path":    None,
            "pending_intent":   None,  # set when we asked for clarification
            "turns":            [],    # [{role, text}] conversation history
        }
    return _sessions[session_id]


def _record_turn(state: dict, role: str, text: str) -> None:
    """Append a turn to this session's history."""
    state["turns"].append({"role": role, "text": text})


# MCP clients — one per server, lazily spawned

_clients: dict[str, MCPClient] = {}

def _get_client(name: str) -> MCPClient:
    """
    Return (lazily creating) the MCPClient for the named server.

    Each client owns exactly one subprocess. We keep them in a dict so we can
    shut them all down cleanly in shutdown(). The server is not spawned until
    the first call(), so a session that never touches vision never starts the
    vision subprocess.
    """
    if name not in _clients:
        scripts = {
            "transcription": "transcription_server.py",
            "vision":        "vision_server.py",
            "summarization": "summarization_server.py",
            "generation":    "generation_server.py",
        }
        script_path = Path(__file__).parent.parent / "mcp_servers" / scripts[name]
        _clients[name] = MCPClient([sys.executable, str(script_path)])
    return _clients[name]


# Clarification helpers


def _clarify(reply: str, options: list[str], pending_intent: str, state: dict) -> dict:
    """
    Return a clarification response and record what we're waiting on.

    The frontend renders options as tappable buttons. On the next turn,
    _resolve_clarification() checks if the user's reply matches one of them.
    """
    state["pending_intent"] = pending_intent
    return {
        "reply":               reply,
        "needs_clarification": True,
        "options":             options,
        "artifact_path":       "",
    }


def _resolve_clarification(state: dict, text: str) -> str | None:
    """
    If the user's text matches one of the pending clarification options
    (or strongly implies one), return that intent. Otherwise return None
    so the caller falls through to fresh classification.

    We do a simple substring check on lowercased text against option keywords.
    This handles both tapped buttons (exact text) and typed replies ("just the pdf").
    """
    if state["pending_intent"] is None:
        return None

    lowered = text.lower()
    state["pending_intent"] = None  # consume the pending intent regardless

    # Map option keywords to intents for resolution
    resolution_map = {
        "transcribe": "transcribe",
        "transcript": "transcribe",
        "objects":    "vision_objects",
        "graph":      "vision_text_graph",
        "summarize":  "summarize",
        "summary":    "summarize",
        "pdf":        "generate_pdf",
        "pptx":       "generate_pptx",
        "powerpoint": "generate_pptx",
        "presentation":"generate_pptx",
        "slides":     "generate_pptx",
    }
    for kw, intent in resolution_map.items():
        if kw in lowered:
            return intent
    return None



# Individual intent handlers
# Each handler takes (state, video_path) and returns a ChatResponse dict.
# They are pure functions of state, no globals beyond the MCP clients.


def _handle_transcribe(state: dict, video_path: str | None) -> dict:
    if not video_path:
        return _reply("Please select a video file first.")

    if state["transcript"] is not None:
        return _reply(f"Transcript (cached):\n\n{state['transcript']}")

    result = _get_client("transcription").call("transcribe", video_path=video_path)
    state["transcript"] = result["text"]
    state["segments"]   = result["segments"]
    return _reply(f"Transcript:\n\n{result['text']}")


def _handle_vision_objects(state: dict, video_path: str | None) -> dict:
    if not video_path:
        return _reply("Please select a video file first.")

    if state["objects"] is not None:
        objs = state["objects"]
    else:
        result = _get_client("vision").call("detect_objects", video_path=video_path)
        state["objects"] = result["objects"]
        objs = state["objects"]

    if not objs:
        return _reply("No objects were detected above the confidence threshold.")

    sorted_objs = sorted(objs, key=lambda o: -o["confidence"])
    lines = [f"  - {o['label']} (confidence {o['confidence']:.2f}, at {o['frame_ts']}s)"
             for o in sorted_objs]
    return _reply("Detected objects:\n" + "\n".join(lines))


def _handle_vision_text_graph(state: dict, video_path: str | None) -> dict:
    if not video_path:
        return _reply("Please select a video file first.")

    if state["text_regions"] is not None:
        regions = state["text_regions"]
        has_graph = state["likely_has_graph"]
    else:
        result = _get_client("vision").call("extract_text", video_path=video_path)
        state["text_regions"]     = result["text_regions"]
        state["likely_has_graph"] = result["likely_has_graph"]
        regions   = state["text_regions"]
        has_graph = state["likely_has_graph"]

    graph_line = "Yes, the video likely contains graphs or charts." if has_graph \
                 else "No graphs or charts were detected."
    if regions:
        sample = regions[:10]
        text_line = "Sample text visible in the video:\n" + "\n".join(f"  - {t}" for t in sample)
        if len(regions) > 10:
            text_line += f"\n  ... and {len(regions) - 10} more text regions."
    else:
        text_line = "No text was detected in the video frames."

    return _reply(f"{graph_line}\n\n{text_line}")


def _handle_summarize(state: dict, video_path: str | None, mode: str = "brief") -> dict:
    if not video_path:
        return _reply("Please select a video file first.")

    # Summarization requires a transcript. Get one (from cache or fresh).
    if state["transcript"] is None:
        trans_result = _get_client("transcription").call("transcribe", video_path=video_path)
        state["transcript"] = trans_result["text"]
        state["segments"]   = trans_result["segments"]

    if not state["transcript"].strip():
        return _reply("The transcript appears to be empty — no speech was detected.")

    # Use cached summary if available and mode matches.
    if state["summary"] is not None:
        s = state["summary"]
    else:
        s = _get_client("summarization").call(
            "summarize", text=state["transcript"], mode=mode
        )
        state["summary"] = s

    key_points = s.get("key_points", [])
    summary    = s.get("summary", "")

    kp_lines = "\n".join(f"  {i+1}. {pt}" for i, pt in enumerate(key_points))
    return _reply(f"Key points:\n{kp_lines}\n\nSummary:\n{summary}")


def _handle_generate(state: dict, video_path: str | None, fmt: str) -> dict:
    """
    Multi-step chain: transcribe (if needed) -> summarize (if needed) -> generate file.

    fmt is "pdf" or "pptx".
    """
    if not video_path:
        return _reply("Please select a video file first.")

    # Step 1: ensure we have a transcript.
    if state["transcript"] is None:
        trans_result = _get_client("transcription").call("transcribe", video_path=video_path)
        state["transcript"] = trans_result["text"]
        state["segments"]   = trans_result["segments"]

    if not state["transcript"].strip():
        return _reply("The transcript appears to be empty — no speech was detected.")

    # Step 2: ensure we have a summary.
    if state["summary"] is None:
        state["summary"] = _get_client("summarization").call(
            "summarize", text=state["transcript"], mode="detailed"
        )

    s = state["summary"]

    # Step 3: assemble structured content for the generation server.
    # The generation server only knows how to render; we decide what goes in.
    sections = [
        {
            "heading": "Key Points",
            "bullets": s.get("key_points", []),
        },
        {
            "heading": "Summary",
            "bullets": [s.get("summary", "")],
        },
    ]

    # Include detected objects if we have them (they add useful context).
    if state["objects"]:
        obj_bullets = [
            f"{o['label']} (confidence {o['confidence']:.2f})"
            for o in sorted(state["objects"], key=lambda o: -o["confidence"])
        ]
        sections.append({"heading": "Detected Objects", "bullets": obj_bullets})

    # Step 4: call the generation server.
    title = Path(video_path).stem.replace("_", " ").replace("-", " ").title()
    if fmt == "pdf":
        result = _get_client("generation").call("make_pdf", title=title, sections=sections)
    else:
        # PPTX uses "slides" not "sections" — same shape, different key name.
        result = _get_client("generation").call("make_pptx", title=title, slides=sections)

    out_path = result["path"]
    state["artifact_path"] = out_path

    fmt_label = "PDF" if fmt == "pdf" else "PowerPoint presentation"
    return {
        "reply":               f"Your {fmt_label} has been generated.",
        "needs_clarification": False,
        "options":             [],
        "artifact_path":       out_path,
    }


# Public API

def handle(session_id: str, text: str, video_path: str) -> dict:
    """
    Process one chat turn. Called by the gRPC server for every SendMessage RPC.

    Returns a dict matching the ChatResponse proto fields:
      reply               str   — text to display in the chat bubble
      needs_clarification bool  — True if we need the user to pick an option
      options             list  — clarification choices (empty if not clarifying)
      artifact_path       str   — path to generated file, or "" if none

    Never raises. All exceptions become error reply strings so the gRPC server
    always has a valid response to send.
    """
    state = _get_session(session_id)

    # Update video path if one was provided this turn.
    if video_path:
        # If the video path changed, invalidate cached inference results — they
        # belong to the old video, not the new one.
        if state["video_path"] and state["video_path"] != video_path:
            _invalidate_cache(state)
        state["video_path"] = video_path

    effective_video = state["video_path"]

    # Record the user's turn in conversation history.
    _record_turn(state, "user", text)

    try:
        response = _dispatch(state, text, effective_video)
    except Exception as exc:
        response = _reply(f"Sorry, something went wrong: {exc}")

    # Record the assistant's reply.
    _record_turn(state, "assistant", response["reply"])
    return response


def _dispatch(state: dict, text: str, video_path: str | None) -> dict:
    """
    Core routing logic. Separated from handle() so exceptions bubble up cleanly.

    Flow:
      1. Check if we're resolving a pending clarification from last turn.
      2. Classify the current message.
      3. If ambiguous -> ask for clarification.
      4. Otherwise -> route to the appropriate handler.
    """
    # Step 1: clarification resolution.
    # If the last response asked the user to choose, check their reply first.
    resolved_intent = _resolve_clarification(state, text)
    if resolved_intent:
        return _route(state, resolved_intent, video_path)

    # Step 2: fresh classification.
    scores = _classify(text)
    intent, _ = _pick_intent(scores)

    # Step 3: ambiguous -> ask.
    if intent == "ambiguous":
        return _build_clarification(scores, state)

    # Step 4: route.
    return _route(state, intent, video_path)


def _route(state: dict, intent: str, video_path: str | None) -> dict:
    """Map a resolved intent to the appropriate handler."""
    if intent == "transcribe":
        return _handle_transcribe(state, video_path)
    elif intent == "vision_objects":
        return _handle_vision_objects(state, video_path)
    elif intent == "vision_text_graph":
        return _handle_vision_text_graph(state, video_path)
    elif intent == "summarize":
        return _handle_summarize(state, video_path)
    elif intent == "generate_pdf":
        return _handle_generate(state, video_path, fmt="pdf")
    elif intent == "generate_pptx":
        return _handle_generate(state, video_path, fmt="pptx")
    else:
        return _reply(
            "I'm not sure what you'd like me to do. Try: "
            "'transcribe the video', 'what objects appear', "
            "'are there graphs', 'summarize', 'make a PDF', or 'make a PowerPoint'."
        )


def _build_clarification(scores: list[tuple[str, float]], state: dict) -> dict:
    """
    Build a clarification response when intent is ambiguous.

    If we have some scoring intents, surface the top ones as options.
    If we have nothing at all, give the user a menu of all capabilities.
    """
    # Human-readable labels for intent names.
    labels = {
        "transcribe":        "Transcribe the video",
        "vision_objects":    "Detect objects in the video",
        "vision_text_graph": "Detect graphs or text in the video",
        "summarize":         "Summarize the video",
        "generate_pdf":      "Generate a PDF report",
        "generate_pptx":     "Generate a PowerPoint presentation",
    }

    if scores:
        # Surface up to 3 top-scoring intents as options.
        options = [labels[intent] for intent, _ in scores[:3] if intent in labels]
        reply = "I'm not quite sure what you'd like. Did you mean:"
    else:
        # No match at all — give full menu.
        options = list(labels.values())
        reply = "I can help with any of the following:"

    return _clarify(reply, options, pending_intent="ambiguous", state=state)


# Cache invalidation

def _invalidate_cache(state: dict) -> None:
    """
    Clear all cached inference results. Called when the user switches to a
    different video, results from the old video must not bleed into the new one.
    """
    state["transcript"]       = None
    state["segments"]         = None
    state["objects"]          = None
    state["text_regions"]     = None
    state["likely_has_graph"] = None
    state["summary"]          = None
    state["artifact_path"]    = None
    # Note: we do NOT clear turns, conversation history persists across video switches.


# Helpers

def _reply(text: str) -> dict:
    """Construct a plain (non-clarification) ChatResponse-shaped dict."""
    return {
        "reply":               text,
        "needs_clarification": False,
        "options":             [],
        "artifact_path":       "",
    }


def shutdown() -> None:
    """Shut down all MCP server subprocesses. Call before the gRPC server exits."""
    for client in _clients.values():
        client.shutdown()
    _clients.clear()
