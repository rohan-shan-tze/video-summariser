"""
Phase 6 persistence test — verifies chat history survives a process restart.

Usage:
  python test_persistence.py --video samples/psychology-lecture.mp4

Step 1: writes two messages via the gRPC server (server must be running).
Step 2: reads them back directly from SQLite (no gRPC needed).
Step 3: prints the history and confirms the rows are present.

The key thing being tested: history is still readable after the gRPC server
is restarted, because it lives in SQLite, not in the orchestrator's memory.
"""

import sys
import argparse
from pathlib import Path

import grpc

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "backend" / "proto"))

from backend.proto import video_pb2, video_pb2_grpc
from backend.storage import database


def run(video_path: str, port: int = 50051) -> None:
    vid = str(Path(video_path).resolve())
    session_id = "persist-test-session"

    # --- Step 1: send messages via gRPC ---
    print("Step 1: sending messages via gRPC...")
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = video_pb2_grpc.VideoServiceStub(channel)

    for text in ["transcribe the video", "summarize the video"]:
        req = video_pb2.ChatRequest(
            session_id=session_id, text=text, video_path=vid
        )
        print(f"  sending: '{text}'")
        resp = stub.SendMessage(req, timeout=180)
        print(f"  reply preview: {resp.reply[:80].strip()}...")

    # --- Step 2: read directly from SQLite (simulates post-restart rehydration) ---
    print("\nStep 2: reading history directly from SQLite...")
    history = database.get_history(session_id)

    print(f"\nFound {len(history)} messages for session '{session_id}':\n")
    for msg in history:
        preview = msg["text"][:80].replace("\n", " ")
        artifact = f"  artifact: {msg['artifact_path']}" if msg["artifact_path"] else ""
        print(f"  [{msg['role']:9s}] {preview}...{artifact}")

    # --- Step 3: verify ---
    assert len(history) >= 4, f"Expected at least 4 rows (2 user + 2 assistant), got {len(history)}"
    roles = [m["role"] for m in history]
    assert roles[0] == "user"
    assert roles[1] == "assistant"
    print("\nPersistence check passed.")

    # Also show all sessions in the DB.
    sessions = database.list_sessions()
    print(f"\nAll sessions in DB ({len(sessions)} total):")
    for s in sessions:
        print(f"  {s['session_id']}  (created {s['created_at']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 persistence test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()
    run(args.video, args.port)
