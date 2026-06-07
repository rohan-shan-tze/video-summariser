"""
Phase 8 - end-to-end test of the five spec queries from the assignment.

The gRPC server must already be running:
  python backend/grpc_server.py

Usage:
  python test_spec_queries.py --video samples/psychology-lecture.mp4

Queries tested (verbatim from the assignment spec):
  1. "Transcribe the video."
  2. "Create a PowerPoint with the key points discussed in the video."
  3. "What objects are shown in the video?"
  4. "Are there any graphs in the video? If yes, describe them."
  5. "Summarize our discussion so far and generate a PDF."
"""

import sys
import argparse
from pathlib import Path

import grpc

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "backend" / "proto"))

from backend.proto import video_pb2, video_pb2_grpc


def send(stub, session_id, text, video_path="", timeout=300):
    req = video_pb2.ChatRequest(
        session_id=session_id,
        text=text,
        video_path=video_path,
    )
    print(f"\n{'='*60}")
    print(f"QUERY: {text}")
    print(f"{'='*60}")
    resp = stub.SendMessage(req, timeout=timeout)
    print(f"REPLY:\n{resp.reply}")
    if resp.needs_clarification:
        print(f"CLARIFICATION OPTIONS: {list(resp.options)}")
    if resp.artifact_path:
        p = Path(resp.artifact_path)
        size = p.stat().st_size if p.exists() else 0
        print(f"ARTIFACT: {resp.artifact_path} ({'OK' if p.exists() else 'MISSING'}, {size:,} bytes)")
    return resp


def run(video_path, port=50051):
    vid = str(Path(video_path).resolve())
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = video_pb2_grpc.VideoServiceStub(channel)

    # All five queries use the same session so query 5 ("our discussion so far")
    # can reference results from queries 1-4.
    session = "spec-query-session"

    print(f"\nRunning 5 spec queries on: {vid}")
    print(f"Session: {session}\n")

    r1 = send(stub, session, "Transcribe the video.", vid, timeout=180)
    r2 = send(stub, session, "Create a PowerPoint with the key points discussed in the video.", timeout=120)
    r3 = send(stub, session, "What objects are shown in the video?", timeout=180)
    r4 = send(stub, session, "Are there any graphs in the video? If yes, describe them.", timeout=300)
    r5 = send(stub, session, "Summarize our discussion so far and generate a PDF.", timeout=120)

    print("\n\n--- RESULTS SUMMARY ---")
    checks = [
        ("1. Transcribe",  "transcript" in r1.reply.lower() or len(r1.reply) > 100),
        ("2. PPTX",        bool(r2.artifact_path) and Path(r2.artifact_path).exists()),
        ("3. Objects",     "person" in r3.reply.lower() or "object" in r3.reply.lower()),
        ("4. Graphs",      "graph" in r4.reply.lower() or "chart" in r4.reply.lower() or "detected" in r4.reply.lower()),
        ("5. PDF",         bool(r5.artifact_path) and Path(r5.artifact_path).exists()),
    ]
    all_pass = True
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  {label}")

    print(f"\n{'ALL QUERIES PASSED' if all_pass else 'SOME QUERIES FAILED'}")
    return all_pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    ok = run(args.video, args.port)
    sys.exit(0 if ok else 1)
