"""Run a self-play cabt game with the Lucario agent.

Usage:
    python build_deck.py    # ensure deck.csv exists (from lucario_deck.csv)
    python run_local.py

Requires kaggle-environments with cabt (Linux/Kaggle — not macOS).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from agent import agent
from build_deck import DECK_OUT, LUCARIO_DECK, load_ids, read_ids, write_deck

HERE = Path(__file__).parent


def _require_cabt() -> None:
    from kaggle_environments import environments

    if "cabt" not in environments:
        print(
            "cabt engine is not available in this environment.\n"
            "\n"
            "The cabt simulator ships a Linux-only native library (libcg.so). "
            "Run on Kaggle or Linux (e.g. Docker --platform linux/amd64).\n",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_deck() -> list[int]:
    if not DECK_OUT.exists():
        write_deck(load_ids(LUCARIO_DECK))
    deck = read_ids(DECK_OUT)
    if len(deck) != 60:
        raise SystemExit(f"Expected 60 cards in {DECK_OUT}, got {len(deck)}")
    return deck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vis",
        type=Path,
        default=HERE / "vis.json",
        help="Write step visualization JSON here",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Write HTML replay here (needs cabt.js)",
    )
    args = parser.parse_args()

    _require_cabt()

    from kaggle_environments import make

    deck = load_deck()
    print("Deck size:", len(deck))
    print("Counts:", Counter(deck))
    print("Deck:", deck)
    deck_hash = hashlib.md5(str(deck).encode()).hexdigest()[:8]
    print(f"Deck hash: {deck_hash}  Timestamp: {time.time()}")

    env = make("cabt", configuration={"decks": [deck.copy(), deck.copy()]}, debug=True)
    env.run([agent, agent])

    print(f"\nSteps played: {len(env.steps)}")
    for i, agent_state in enumerate(env.state):
        print(
            f"Player {i}: status={agent_state['status']}, "
            f"reward={agent_state.get('reward')}"
        )
        if agent_state.get("error"):
            print(f"  error: {agent_state['error']}")

    if env.steps:
        try:
            visualize = env.steps[0][0]["visualize"]
        except (KeyError, TypeError, IndexError):
            visualize = None
        if visualize is not None:
            args.vis.write_text(json.dumps(visualize, indent=2, default=str))
            print(f"Visualization written to {args.vis}")
        else:
            print("(No visualize data on env.steps[0][0])")

    if args.replay:
        try:
            args.replay.write_text(env.render(mode="html"))
            print(f"HTML replay written to {args.replay}")
        except FileNotFoundError as exc:
            print(f"(HTML render unavailable: {exc.filename} missing)")

    print("Simulation finished.")


if __name__ == "__main__":
    main()
