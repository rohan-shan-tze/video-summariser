"""
MCP base server stdio JSON-RPC transport.

How it works:
  Each MCP server is a standalone process. The orchestrator spawns it as a
  subprocess and communicates via its stdin/stdout using newline-delimited JSON.
  This is the MCP stdio transport: one JSON object per line, no framing headers.

Request format  (orchestrator -> server, on stdin):
  {"id": <int>, "method": "tool_name", "params": {...}}

Response format (server -> orchestrator, on stdout):
  {"id": <int>, "result": {...}}          on success
  {"id": <int>, "error": {"message": ""}} on failure

Why stdio JSON-RPC (not HTTP/sockets)?
  - Zero port management. The subprocess IS the server; it vanishes with the parent.
  - MCP spec uses stdio as its canonical transport for local servers.
  - Keeps each server completely isolated: crash one, the rest keep running.
"""

import sys
import json
import traceback
from typing import Any, Callable


class MCPServer:
    """
    Base class for all MCP servers in this project.

    Subclasses call self.register_tool(name, fn) for each capability they expose,
    then call self.run() to enter the read-dispatch-reply loop.
    """

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        # Registry maps tool name (string) -> callable that handles it.
        # Populated by subclasses via register_tool() before run() is called.
        self._tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, name: str, fn: Callable[..., Any]) -> None:
        """Register a callable as an MCP tool reachable by name."""
        self._tools[name] = fn

    # Main loop

    def run(self) -> None:
        """
        Block forever reading one JSON request per line from stdin,
        dispatch to the registered tool, and write one JSON response to stdout.

        stdin/stdout are line-buffered here; stderr is left for debug logging
        so it doesn't contaminate the JSON channel.
        """
        # Flush after every write so the orchestrator doesn't block waiting
        # for a full buffer.
        sys.stdout.reconfigure(line_buffering=True)   # type: ignore[attr-defined]

        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue  # skip blank lines (e.g. keepalive pings)

            response = self._handle(raw_line)
            # Serialize and flush immediately, the orchestrator is blocking on this.
            print(json.dumps(response), flush=True)

    # Internal dispatch

    def _handle(self, raw_line: str) -> dict:
        """Parse one request line and return the response dict."""
        # Step 1: parse JSON. A malformed line gets an error response with id=null
        # because we don't even know the request id yet.
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            return {"id": None, "error": {"message": f"Invalid JSON: {exc}"}}

        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        # Step 2: look up the tool. Unknown method -> error (not a crash).
        if method not in self._tools:
            return {
                "id": req_id,
                "error": {"message": f"Unknown tool '{method}' in {self.server_name}"},
            }

        # Step 3: call the tool. Any exception is caught and returned as an error
        # rather than killing the server process, the orchestrator can handle it.
        try:
            result = self._tools[method](**params)
            return {"id": req_id, "result": result}
        except Exception as exc:
            # Print traceback to stderr (not stdout) so it doesn't break JSON channel
            traceback.print_exc(file=sys.stderr)
            return {"id": req_id, "error": {"message": str(exc)}}
