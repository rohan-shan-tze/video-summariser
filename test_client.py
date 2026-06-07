"""
Standalone test client for Phase 1 checkpoint.

Usage:
  python test_client.py --video samples/sample.mp4
  python test_client.py --video samples/sample.mp4 --port 50051

Sends "transcribe the video" to the running gRPC server and prints the reply.
The gRPC server must already be running:
  python backend/grpc_server.py
"""

import sys
import argparse
from pathlib import Path

import grpc

_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))
# grpc_tools generates stubs with bare `import video_pb2` — proto dir must be on path
sys.path.insert(0, str(_REPO_ROOT / "backend" / "proto"))
from backend.proto import video_pb2, video_pb2_grpc


def run(video_path: str, port: int = 50051) -> None:
    # Connect to localhost only — matches the server's bind address.
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = video_pb2_grpc.VideoServiceStub(channel)

    request = video_pb2.ChatRequest(
        session_id="test-session-1",
        text="transcribe the video",
        video_path=str(Path(video_path).resolve()),
    )

    print(f"[test_client] Sending: '{request.text}' for video: {request.video_path}")
    response = stub.SendMessage(request, timeout=120)  # up to 2 min for first-run model load

    print(f"\n[test_client] needs_clarification: {response.needs_clarification}")
    print(f"[test_client] artifact_path:       '{response.artifact_path}'")
    print(f"\n--- REPLY ---\n{response.reply}\n--- END ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 test client")
    parser.add_argument("--video", required=True, help="Path to a .mp4 file")
    parser.add_argument("--port",  type=int, default=50051)
    args = parser.parse_args()
    run(args.video, args.port)
