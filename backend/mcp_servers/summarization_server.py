"""
Summarization MCP Server.

Exposes one tool:
  summarize(text, mode) -> {key_points: [str], summary: str, method: str}

Two backends are available, selected by the mode parameter:

  LLM modes (default):
    "llm_brief"    - Llama 3.2 1B generates a 3-point summary (concise)
    "llm_detailed" - Llama 3.2 1B generates a 5-point summary (thorough)

  Extractive modes (classical, no model):
    "brief"    - LexRank selects 3 key sentences, 2-sentence summary
    "detailed" - LexRank selects 5 key sentences, 5-sentence summary

The LLM backend uses llama.cpp (via llama-cpp-python) with the Llama 3.2 1B
Q4_K_M GGUF running on CPU only. The model is lazy-loaded on first use and
stays resident in the subprocess for subsequent calls.

Why two backends?
  - LLM: produces natural, synthesised prose; better for a genAI context.
    Slower (~15-30s on CPU), requires ~1.5 GB RAM for the model.
  - Extractive (LexRank): deterministic, instant, zero RAM overhead beyond
    the text itself. Useful when speed matters or as a reliable fallback.

The interface is identical for both - callers see the same {key_points,
summary, method} shape regardless of which backend ran.

Run as a subprocess; communicates via stdio JSON-RPC (see mcp_common/server.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.mcp_common.server import MCPServer


# Paths

_MODELS_DIR  = Path(__file__).parent.parent.parent / "models"
_NLTK_DIR    = _MODELS_DIR / "nltk"
_LLM_DIR     = _MODELS_DIR / "llm"
_LLM_PATH    = _LLM_DIR / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

# Lazy-loaded LLM singleton - loaded once, reused across calls.
_llm = None


# NLTK bootstrap (used by the extractive backend)

def _ensure_nltk():
    """Download NLTK punkt tokenizer data into models/nltk/ if not present."""
    import nltk
    _NLTK_DIR.mkdir(parents=True, exist_ok=True)
    if str(_NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(_NLTK_DIR))
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
        except (LookupError, OSError):
            token = resource.split("/")[1]
            nltk.download(token, download_dir=str(_NLTK_DIR), quiet=True)


# LLM backend

def _get_llm():
    """
    Lazy-load the Llama 3.2 1B GGUF model via llama-cpp-python.

    Parameters chosen for CPU-only inference:
      n_ctx=2048    - context window; enough for a 1-minute transcript plus prompt
      n_threads=4   - use 4 CPU threads; balances speed vs. system responsiveness
      n_gpu_layers=0 - CPU only, never offload to GPU
      verbose=False  - suppress llama.cpp progress bars from stdout
                       (they would break the JSON-RPC channel on stdout)
    """
    global _llm
    if _llm is not None:
        return _llm

    if not _LLM_PATH.exists():
        raise FileNotFoundError(
            f"LLM model not found at {_LLM_PATH}. "
            "Run: python download_models.py"
        )

    from llama_cpp import Llama
    _llm = Llama(
        model_path=str(_LLM_PATH),
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,  # CPU only - never change this in committed code
        verbose=False,
    )
    return _llm


def _summarize_llm(text: str, n_points: int) -> dict:
    """
    Use Llama 3.2 1B to generate a structured summary.

    Prompt design:
      - System message constrains the model to the summarisation task only.
      - User message provides the transcript and asks for exactly N bullet points
        followed by a one-paragraph prose summary.
      - We use the instruct chat template (create_chat_completion) so the model
        applies its instruction-following fine-tuning correctly.
      - max_tokens=512 is enough for N bullet points + a short paragraph.
      - temperature=0.3 keeps output focused and mostly deterministic while
        allowing slightly more natural phrasing than temperature=0.
    """
    llm = _get_llm()

    system_prompt = (
        "You are a precise summarisation assistant. "
        "You summarise transcripts accurately without adding information "
        "that is not present in the source text."
    )

    user_prompt = (
        f"Summarise the following transcript.\n\n"
        f"First, list exactly {n_points} key points as a numbered list "
        f"(one sentence each).\n"
        f"Then write a short paragraph summary (2-3 sentences).\n\n"
        f"Format your response as:\n"
        f"KEY POINTS:\n"
        f"1. ...\n"
        f"2. ...\n"
        f"...\n\n"
        f"SUMMARY:\n"
        f"...\n\n"
        f"Transcript:\n{text}"
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )

    raw = response["choices"][0]["message"]["content"].strip()
    return _parse_llm_output(raw, n_points)


def _parse_llm_output(raw: str, n_points: int) -> dict:
    """
    Parse the structured output from the LLM into {key_points, summary}.

    The model is prompted to produce a rigid format (KEY POINTS: / SUMMARY:).
    We split on those headers. If the model deviates we fall back to treating
    the whole output as a summary with no separate key points.
    """
    key_points = []
    summary    = ""

    # Split on the SUMMARY: header first to isolate the two sections.
    parts = raw.split("SUMMARY:", 1)
    summary_raw = parts[1].strip() if len(parts) > 1 else raw.strip()

    # Parse the KEY POINTS section if it exists.
    kp_raw = parts[0]
    if "KEY POINTS:" in kp_raw:
        kp_raw = kp_raw.split("KEY POINTS:", 1)[1]

    for line in kp_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1. " / "- " / "* " numbering.
        import re
        line = re.sub(r"^[\d]+\.\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        if len(line) > 10:
            key_points.append(line)

    key_points = key_points[:n_points]
    summary    = summary_raw

    # Fallback: if parsing failed entirely, use the raw output as the summary.
    if not key_points and not summary:
        summary = raw

    return {"key_points": key_points, "summary": summary}


# Extractive backend (LexRank)

_MIN_SENTENCE_CHARS = 20


def _summarize_extractive(text: str, n_points: int, n_summary: int) -> dict:
    """
    Extractive summarisation via LexRank (sumy + NLTK).

    LexRank builds a sentence similarity graph (TF-IDF cosine similarity as
    edge weights) then runs PageRank to score each sentence. The top-scoring
    sentences are the most "central" in meaning across the document.
    """
    _ensure_nltk()

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers   import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    parser     = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()

    n_total          = max(n_points, n_summary)
    ranked_sentences = summarizer(parser.document, n_total)

    ranked_texts = [str(s) for s in ranked_sentences
                    if len(str(s)) >= _MIN_SENTENCE_CHARS]

    if not ranked_texts:
        fallback = text.strip()[:500]
        return {"key_points": [fallback], "summary": fallback}

    key_points = ranked_texts[:n_points]

    summary_pool    = ranked_texts[:n_summary]
    all_sentences   = [str(s) for s in parser.document.sentences]
    summary_ordered = _restore_order(summary_pool, all_sentences)
    summary         = " ".join(summary_ordered)

    return {"key_points": key_points, "summary": summary}


def _restore_order(selected: list[str], all_sentences: list[str]) -> list[str]:
    """Return selected sentences sorted by their original document position."""
    position = {s: i for i, s in enumerate(all_sentences)}
    return sorted(selected, key=lambda s: position.get(s, len(all_sentences)))


# Public tool

def summarize(text: str, mode: str = "llm_brief") -> dict:
    """
    Summarize text using the selected backend.

    mode values:
      "llm_brief"    - LLM, 3 key points  (default)
      "llm_detailed" - LLM, 5 key points
      "brief"        - LexRank, 3 key points, 2-sentence summary
      "detailed"     - LexRank, 5 key points, 5-sentence summary

    Returns:
      key_points  list of key point strings
      summary     prose summary string
      method      "llm" or "extractive" - lets callers know which backend ran
    """
    if not text or not text.strip():
        return {"key_points": [], "summary": "", "method": "none"}

    if mode == "llm_brief":
        result = _summarize_llm(text, n_points=3)
        result["method"] = "llm"
        return result

    if mode == "llm_detailed":
        result = _summarize_llm(text, n_points=5)
        result["method"] = "llm"
        return result

    # Extractive modes
    configs = {
        "brief":    {"key_points": 3, "summary_sentences": 2},
        "detailed": {"key_points": 5, "summary_sentences": 5},
    }
    cfg    = configs.get(mode, configs["brief"])
    result = _summarize_extractive(text, cfg["key_points"], cfg["summary_sentences"])
    result["method"] = "extractive"
    return result


# Entrypoint

if __name__ == "__main__":
    server = MCPServer("summarization_server")
    server.register_tool("summarize", summarize)
    server.run()
