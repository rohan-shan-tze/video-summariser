"""
gRPC server entrypoint.

Binds to 127.0.0.1 (localhost only, never 0.0.0.0).
Implements VideoService.SendMessage by forwarding to the orchestrator.

Usage:
  python backend/grpc_server.py [--port 50051]
"""

import sys
import signal
import argparse
from pathlib import Path
from concurrent import futures

import grpc

# Add repo root to path so backend.* imports resolve.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
# grpc_tools generates stubs with bare `import video_pb2`. Add proto dir so it resolves
sys.path.insert(0, str(_REPO_ROOT / "backend" / "proto"))

from backend.proto import video_pb2, video_pb2_grpc
from backend.orchestrator import orchestrator

_DEFAULT_PORT = 50051
# Bind to loopback only, gRPC must never be exposed on 0.0.0.0.
_BIND_ADDRESS = "127.0.0.1"


class VideoServiceServicer(video_pb2_grpc.VideoServiceServicer):
    """
    Implements the VideoService gRPC service defined in video.proto.

    Each incoming SendMessage RPC:
      1. Extracts session_id, text, video_path from the ChatRequest proto.
      2. Delegates to orchestrator.handle() — all routing/state logic lives there.
      3. Packages the orchestrator's dict response into a ChatResponse proto.

    The gRPC servicer is stateless; all session state is owned by the orchestrator.
    """

    def SendMessage(self, request: video_pb2.ChatRequest,
                    context: grpc.ServicerContext) -> video_pb2.ChatResponse:
        response = orchestrator.handle(
            session_id=request.session_id,
            text=request.text,
            video_path=request.video_path,
        )
        return video_pb2.ChatResponse(
            reply=response["reply"],
            needs_clarification=response["needs_clarification"],
            options=response["options"],
            artifact_path=response["artifact_path"],
        )


def serve(port: int = _DEFAULT_PORT) -> None:
    server = grpc.server(
        # ThreadPoolExecutor with 4 workers handles concurrent RPCs.
        # For Phase 1 a single worker would suffice, but 4 costs nothing.
        futures.ThreadPoolExecutor(max_workers=4)
    )
    video_pb2_grpc.add_VideoServiceServicer_to_server(VideoServiceServicer(), server)

    bind_str = f"{_BIND_ADDRESS}:{port}"
    server.add_insecure_port(bind_str)
    server.start()
    print(f"[grpc_server] Listening on {bind_str}", flush=True)

    # Graceful shutdown on SIGINT / SIGTERM so MCP subprocesses are cleaned up.
    def _stop(signum, frame):
        print("[grpc_server] Shutting down...", flush=True)
        orchestrator.shutdown()
        server.stop(grace=2)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Summariser gRPC server")
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()
    serve(args.port)
