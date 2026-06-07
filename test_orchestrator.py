"""
Phase 5 test client — exercises all major intents through the full stack.

The gRPC server must already be running:
  python backend/grpc_server.py

Usage:
  python test_orchestrator.py --video samples/psychology-lecture.mp4

Tests (in order):
  1. Transcribe
  2. Detect objects
  3. Detect graphs / text
  4. Summarize
  5. Generate PDF  (multi-step chain: transcribe -> summarize -> pdf)
  6. Generate PPTX (reuses cached transcript + summary)
  7. Ambiguous query -> clarification response
  8. Resolve clarification by replying with one of the options
"""

import sys
import argparse
import time
from pathlib import Path

import grpc

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "backend" / "proto"))
from backend.proto import video_pb2, video_pb2_grpc


def send(stub, session_id: str, text: str, video_path: str = "", timeout: int = 180):
    req = video_pb2.ChatRequest(
        session_id=session_id,
        text=text,
        video_path=video_path,
    )
    print(f"\n{'='*60}")
    print(f"USER: {text}")
    if video_path:
        print(f"      (video: {Path(video_path).name})")
    print(f"{'='*60}")

    resp = stub.SendMessage(req, timeout=timeout)

    print(f"REPLY:\n{resp.reply}")
    if resp.needs_clarification:
        print(f"CLARIFICATION OPTIONS: {list(resp.options)}")
    if resp.artifact_path:
        p = Path(resp.artifact_path)
        exists = p.exists()
        size   = p.stat().st_size if exists else 0
        print(f"ARTIFACT: {resp.artifact_path}  ({'OK' if exists else 'MISSING'}, {size:,} bytes)")
    return resp


def run(video_path: str, port: int = 50051):
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub    = video_pb2_grpc.VideoServiceStub(channel)
    vid     = str(Path(video_path).resolve())

    # Each test uses a fresh session_id so they are independent.
    ts = int(time.time())

    print("\n--- Test 1: Transcribe ---")
    send(stub, f"sess-transcribe-{ts}", "transcribe the video", vid, timeout=180)

    print("\n--- Test 2: Detect objects ---")
    send(stub, f"sess-objects-{ts}", "what objects appear in the video?", vid, timeout=120)

    print("\n--- Test 3: Detect graphs/text ---")
    send(stub, f"sess-graphs-{ts}", "are there any graphs or charts in the video?", vid, timeout=300)

    print("\n--- Test 4: Summarize ---")
    send(stub, f"sess-summarize-{ts}", "summarize the video", vid, timeout=180)

    print("\n--- Test 5: Generate PDF (multi-step chain) ---")
    send(stub, f"sess-pdf-{ts}", "make a PDF report", vid, timeout=240)

    print("\n--- Test 6: Generate PPTX (reuses cache from same session) ---")
    sess_pptx = f"sess-pptx-{ts}"
    # First transcribe so cache is warm, then generate PPTX
    send(stub, sess_pptx, "summarize the video", vid, timeout=180)
    send(stub, sess_pptx, "now make a PowerPoint presentation", timeout=60)

    print("\n--- Test 7: Ambiguous query -> clarification ---")
    resp = send(stub, f"sess-ambig-{ts}", "can you help me with the video?", vid)
    assert resp.needs_clarification, "Expected clarification for ambiguous query"

    print("\n--- Test 8: Resolve clarification ---")
    if resp.options:
        # Reply with the first offered option to resolve the clarification
        send(stub, f"sess-ambig-{ts}", resp.options[0], timeout=180)

    print("\n\nAll tests complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 orchestrator test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--port",  type=int, default=50051)
    args = parser.parse_args()
    run(args.video, args.port)
