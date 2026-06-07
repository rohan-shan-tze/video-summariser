"""
Summarization MCP Server.

Exposes one tool:
  summarize(text, mode) -> {key_points: [str], summary: str}

Implementation is fully extractive — no model download, no network calls.
Uses sumy (sentence ranking) backed by the LexRank algorithm, with NLTK
for sentence tokenization.

Why extractive?
  Extractive summarization ranks and selects sentences from the source text
  rather than generating new ones. This means:
    - No model to download, no GPU/CPU inference cost.
    - Output is always grammatically correct (it's real sentences from the input).
    - Fully deterministic and debuggable.
    - Interface is clean: a generative LLM could replace it later without
      changing any caller — just swap the body of summarize().

mode parameter:
  "brief"    -> 3 key points, 2-sentence summary
  "detailed" -> 5 key points, 5-sentence summary
  (defaults to "brief" if unrecognised)

Run as a subprocess; communicates via stdio JSON-RPC (see mcp_common/server.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.mcp_common.server import MCPServer

# NLTK bootstrap

# sumy's sentence tokenizer uses NLTK's punkt model. We download it once into
# a local directory so the server works offline after first setup.

_NLTK_DIR = Path(__file__).parent.parent.parent / "models" / "nltk"


def _ensure_nltk():
    """Download NLTK punkt tokenizer data if not already present."""
    import nltk
    _NLTK_DIR.mkdir(parents=True, exist_ok=True)
    # Prepend our local dir so nltk.data.find() checks it first.
    if str(_NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(_NLTK_DIR))
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
        except (LookupError, OSError):
            # LookupError = not found in any path; OSError = path exists but wrong layout.
            token = resource.split("/")[1]
            nltk.download(token, download_dir=str(_NLTK_DIR), quiet=True)


# Mode config

_MODE_CONFIG = {
    "brief":    {"key_points": 3, "summary_sentences": 2},
    "detailed": {"key_points": 5, "summary_sentences": 5},
}
_DEFAULT_MODE = "brief"

# Minimum characters for a line to be treated as a real sentence (filters
# out very short OCR fragments like "Fig 1" or stray punctuation).
_MIN_SENTENCE_CHARS = 20

# Core summarize logic

def summarize(text: str, mode: str = "brief") -> dict:
    """
    Summarize text using extractive sentence ranking (LexRank via sumy).

    Args:
      text  Full transcript or document text to summarize.
      mode  "brief" (3 key points, 2-sentence summary) or
            "detailed" (5 key points, 5-sentence summary).

    Returns:
      key_points  List of the top-ranked sentences as bullet-point facts.
      summary     Continuous prose: top N sentences joined in their original
                  document order (reads more naturally than ranking order).
    """
    if not text or not text.strip():
        return {"key_points": [], "summary": ""}

    _ensure_nltk()

    cfg = _MODE_CONFIG.get(mode, _MODE_CONFIG[_DEFAULT_MODE])
    n_points   = cfg["key_points"]
    n_summary  = cfg["summary_sentences"]

    # sumy imports are deferred so the server process starts quickly and any
    # missing-package errors surface as MCP error responses.
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    # Parse the raw text into sumy's internal sentence/word representation.
    parser = PlaintextParser.from_string(text, Tokenizer("english"))

    # LexRank builds a graph where nodes are sentences and edge weights are
    # cosine similarity of TF-IDF vectors. It then runs PageRank over that
    # graph. The highest-scoring sentences are the most "central" in meaning.
    # This is robust for lecture/narration text, better than pure frequency
    # (Luhn) because it captures thematic coherence, not just word counts.
    summarizer = LexRankSummarizer()

    # Ask for enough sentences to cover both key_points and summary needs,
    # then split. We take the larger count so we have enough to work with.
    n_total = max(n_points, n_summary)
    ranked_sentences = summarizer(parser.document, n_total)

    # ranked_sentences is in ranking order (most important first).
    # Convert sumy Sentence objects to plain strings.
    ranked_texts = [str(s) for s in ranked_sentences
                    if len(str(s)) >= _MIN_SENTENCE_CHARS]

    if not ranked_texts:
        # Fallback: input was too short for LexRank to produce anything useful.
        # Return the raw text trimmed to a reasonable length.
        fallback = text.strip()[:500]
        return {"key_points": [fallback], "summary": fallback}

    # Key points: top N sentences in ranking order (importance order).
    key_points = ranked_texts[:n_points]

    # Summary: top N sentences sorted back into their original document order
    # so the prose flows naturally rather than jumping around in time.
    summary_pool = ranked_texts[:n_summary]
    # Reconstruct original order by matching against the full sentence list.
    all_sentences = [str(s) for s in parser.document.sentences]
    summary_ordered = _restore_order(summary_pool, all_sentences)
    summary = " ".join(summary_ordered)

    return {
        "key_points": key_points,
        "summary":    summary,
    }


def _restore_order(selected: list[str], all_sentences: list[str]) -> list[str]:
    """
    Given a subset of sentences, return them sorted by their position in the
    original document. This makes the summary read as natural prose.
    """
    # Build an index: sentence text -> first position it appears at.
    position = {s: i for i, s in enumerate(all_sentences)}
    # Sort selected sentences by their original position; unknown sentences
    # (shouldn't happen) go to the end.
    return sorted(selected, key=lambda s: position.get(s, len(all_sentences)))


# Entrypoint

if __name__ == "__main__":
    server = MCPServer("summarization_server")
    server.register_tool("summarize", summarize)
    server.run()
