"""
MCP base client, spawns and talks to a stdio JSON-RPC MCP server subprocess.

How it works:
  The orchestrator creates one MCPClient per MCP server it needs. On first use,
  the client spawns the server as a subprocess with stdin=PIPE, stdout=PIPE.
  Each call() sends one JSON request line to the subprocess's stdin and reads
  one JSON response line from its stdout. This is synchronous (blocking) per
  call, which is fine for our single-threaded orchestrator.

Why spawn on first use ("lazy spawn")?
  We only pay the startup cost for servers that are actually needed in a session.
  A "summarize" query never starts the Vision server.

Thread safety: not thread-safe by design. The orchestrator is single-threaded;
  if we ever parallelize, callers must serialize access per client instance.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Any


class MCPClient:
    """
    Client that owns one MCP server subprocess.

    Usage:
        client = MCPClient([sys.executable, "backend/mcp_servers/transcription_server.py"])
        result = client.call("transcribe", video_path="/path/to/file.mp4")
        client.shutdown()
    """

    def __init__(self, server_cmd: list[str]) -> None:
        """
        server_cmd — the command to launch the MCP server, e.g.:
            [sys.executable, "backend/mcp_servers/transcription_server.py"]
        """
        self._cmd = server_cmd
        self._proc: subprocess.Popen | None = None
        # Monotonically increasing request counter so we can match responses to requests.
        # In this synchronous design we only ever have one in-flight request, but the
        # id field is part of the JSON-RPC protocol and needed for correctness.
        self._next_id = 1

    # Lifecycle

    def _ensure_started(self) -> None:
        """Lazily spawn the server subprocess on first call."""
        if self._proc is not None and self._proc.poll() is None:
            return  # already running

        # Launch with text-mode pipes. stderr inherits the parent's stderr so
        # server-side tracebacks appear in the terminal during development.
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,          # server debug output visible in terminal
            text=True,
            bufsize=1,                  # line-buffered, matches server's reconfigure()
            encoding="utf-8",
        )

    def shutdown(self) -> None:
        """Terminate the server subprocess cleanly."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()    # type: ignore[union-attr]
            self._proc.wait(timeout=5)

    # RPC

    def call(self, method: str, **params: Any) -> dict:
        """
        Send one JSON-RPC request and return the result dict.

        Raises RuntimeError if the server returns an error or if the subprocess
        dies unexpectedly — the orchestrator catches these and turns them into
        user-facing error replies.
        """
        self._ensure_started()

        req_id = self._next_id
        self._next_id += 1

        # Serialise the request as a single line (no embedded newlines).
        request_line = json.dumps({"id": req_id, "method": method, "params": params})

        # Write to the server's stdin, flush immediately so it doesn't block.
        self._proc.stdin.write(request_line + "\n")   # type: ignore[union-attr]
        self._proc.stdin.flush()                       # type: ignore[union-attr]

        # Read exactly one response line from stdout.
        # readline() blocks until the server writes a newline-terminated response.
        raw = self._proc.stdout.readline()             # type: ignore[union-attr]
        if not raw:
            raise RuntimeError(
                f"MCP server '{self._cmd}' closed stdout unexpectedly "
                f"(process exit code: {self._proc.poll()})"
            )

        response = json.loads(raw)

        # Surface server-side errors as Python exceptions.
        if "error" in response:
            raise RuntimeError(f"MCP error from {self._cmd}: {response['error']['message']}")

        return response.get("result", {})
