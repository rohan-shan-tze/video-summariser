"""
Direct test for Phase 2 Vision MCP server (no gRPC — calls tools directly).

Usage:
  python test_vision.py --video samples/psychology-lecture.mp4

Tests both vision tools and prints results.
"""

import sys
import io
import argparse
from pathlib import Path

# EasyOCR prints Unicode progress characters; force UTF-8 on Windows consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from backend.mcp_servers.vision_server import detect_objects, extract_text


def run(video_path: str) -> None:
    path = str(Path(video_path).resolve())
    print(f"Testing vision tools on: {path}\n")

    print("--- detect_objects ---")
    result = detect_objects(path)
    objects = result["objects"]
    if objects:
        for obj in sorted(objects, key=lambda x: -x["confidence"]):
            print(f"  {obj['label']:20s}  conf={obj['confidence']:.3f}  ts={obj['frame_ts']}s")
    else:
        print("  (no objects detected above threshold)")

    print("\n--- extract_text ---")
    result = extract_text(path)
    print(f"  likely_has_graph: {result['likely_has_graph']}")
    regions = result["text_regions"]
    print(f"  text_regions ({len(regions)} unique):")
    for t in regions[:20]:   # cap at 20 for readability
        print(f"    {t!r}")
    if len(regions) > 20:
        print(f"    ... and {len(regions) - 20} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 vision test")
    parser.add_argument("--video", required=True, help="Path to a .mp4 file")
    args = parser.parse_args()
    run(args.video)
