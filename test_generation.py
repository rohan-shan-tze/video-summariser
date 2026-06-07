"""
Direct test for Phase 4 Generation MCP server (no gRPC, calls tools directly).

Usage:
  python test_generation.py

Generates a sample PDF and PPTX into the outputs/ directory and prints the paths.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.mcp_servers.generation_server import make_pdf, make_pptx

SAMPLE_TITLE = "Psychology Lecture: Introduction to Cognitive Biases"

SAMPLE_SECTIONS = [
    {
        "heading": "Key Points",
        "bullets": [
            "Cognitive biases are systematic errors in thinking that affect decisions.",
            "Confirmation bias leads people to favor information that confirms existing beliefs.",
            "The availability heuristic causes people to overestimate the likelihood of vivid events.",
            "Anchoring bias means first impressions disproportionately influence final judgements.",
        ],
    },
    {
        "heading": "Summary",
        "bullets": [
            "This lecture introduced the concept of cognitive biases and their impact on human decision-making.",
            "Three major biases were discussed: confirmation bias, availability heuristic, and anchoring.",
        ],
    },
    {
        "heading": "Detected Objects",
        "bullets": [
            "person (confidence: 0.888)",
            "laptop (confidence: 0.442)",
            "tie (confidence: 0.674)",
        ],
    },
]

SAMPLE_SLIDES = [
    {
        "heading": "What Are Cognitive Biases?",
        "bullets": [
            "Systematic errors in thinking that affect decisions and judgements",
            "Arise from mental shortcuts (heuristics) the brain uses to process information",
            "Universal — affect everyone regardless of intelligence",
        ],
    },
    {
        "heading": "Confirmation Bias",
        "bullets": [
            "Tendency to search for information that confirms existing beliefs",
            "Leads to ignoring contradictory evidence",
            "Common in politics, science, and everyday decisions",
        ],
    },
    {
        "heading": "Availability Heuristic",
        "bullets": [
            "Judging probability by how easily examples come to mind",
            "Vivid or recent events feel more likely than statistics suggest",
            "Example: overestimating plane crash risk after seeing news coverage",
        ],
    },
    {
        "heading": "Anchoring Bias",
        "bullets": [
            "First piece of information seen disproportionately influences judgement",
            "Used heavily in negotiation and pricing strategies",
            "Hard to overcome even when we know about it",
        ],
    },
]


if __name__ == "__main__":
    print("Generating PDF...")
    pdf_result = make_pdf(title=SAMPLE_TITLE, sections=SAMPLE_SECTIONS)
    print(f"  PDF written to: {pdf_result['path']}")

    print("Generating PPTX...")
    pptx_result = make_pptx(title=SAMPLE_TITLE, slides=SAMPLE_SLIDES)
    print(f"  PPTX written to: {pptx_result['path']}")

    # Verify both files exist and are non-empty
    for label, path_str in [("PDF", pdf_result["path"]), ("PPTX", pptx_result["path"])]:
        p = Path(path_str)
        if p.exists() and p.stat().st_size > 0:
            print(f"  {label}: OK ({p.stat().st_size:,} bytes)")
        else:
            print(f"  {label}: FAILED — file missing or empty")
