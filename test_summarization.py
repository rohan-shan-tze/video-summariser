"""
Direct test for Phase 3 Summarization MCP server (no gRPC — calls tool directly).

Usage:
  python test_summarization.py
  python test_summarization.py --video samples/psychology-lecture.mp4

With --video: transcribes the video first (requires the gRPC server to be running),
then summarizes the result. Without --video: runs on a built-in sample text.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.mcp_servers.summarization_server import summarize


SAMPLE_TEXT = """
Reinforcement learning is a type of machine learning where an agent learns to make
decisions by interacting with an environment. The agent receives rewards for good actions
and penalties for bad ones. Over time, the agent learns a policy that maximizes cumulative
reward. This is similar to how humans learn through trial and error. Deep reinforcement
learning combines reinforcement learning with deep neural networks, allowing agents to
handle high-dimensional state spaces such as raw pixel inputs from video games.
One famous example is AlphaGo, which defeated the world champion in the game of Go.
The key challenge in reinforcement learning is the exploration-exploitation tradeoff:
the agent must balance trying new actions to discover better strategies versus using
known good strategies to collect rewards. Temporal difference learning is a method
that updates value estimates based on the difference between predicted and actual rewards.
The Q-learning algorithm is a specific form of temporal difference learning that learns
the value of taking each action in each state. Policy gradient methods directly optimize
the policy without explicitly computing value functions, and have been successful in
continuous action spaces such as robotic control.
"""


def run(text: str, label: str) -> None:
    print(f"=== Source: {label} ===")
    print(f"Input length: {len(text)} characters\n")

    for mode in ("brief", "detailed"):
        print(f"--- mode={mode} ---")
        result = summarize(text, mode=mode)

        print("Key points:")
        for i, point in enumerate(result["key_points"], 1):
            print(f"  {i}. {point}")

        print(f"\nSummary:\n  {result['summary']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 summarization test")
    parser.add_argument("--video", help="Optional: path to .mp4 to transcribe first")
    args = parser.parse_args()

    if args.video:
        # Transcribe via the transcription server directly (same pattern as test_vision.py).
        from backend.mcp_servers.transcription_server import transcribe
        print(f"Transcribing {args.video} ...\n")
        result = transcribe(str(Path(args.video).resolve()))
        text = result["text"]
        run(text, label=args.video)
    else:
        run(SAMPLE_TEXT.strip(), label="built-in sample text")
