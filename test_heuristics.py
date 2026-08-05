"""Smoke-test every rule-based heuristic agent under heuristics/.

Each heuristic plays a full self-play battle against itself (its own
deck.csv on both sides) through the same kaggle_environments `cabt` harness
local_test.py uses. A heuristic "works" if: its deck.csv has exactly 60
cards, the engine accepts it, and the battle runs to completion with no
error/timeout/invalid status on either side.

The RL agent (agent.py / ppo_crustle.zip) is intentionally excluded -- it's
not a heuristic, and is currently broken (obs_vectorizer/model dimension
mismatch, see agent.py).

Usage:
    python test_heuristics.py
"""

import sys

from kaggle_environments import make

from heuristics.crustle_agent import agent as crustle_agent
from heuristics.abomasnow_agent import agent as abomasnow_agent
from heuristics.dragapult_agent import agent as dragapult_agent
from heuristics.dragapult_v2_agent import agent as dragapult_v2_agent
from heuristics.iono_agent import agent as iono_agent
from heuristics.archaludon_agent import agent as archaludon_agent
from heuristics.ragingbolt_agent import agent as ragingbolt_agent
import heuristics.ragingbolt_agent.ragingbolt_agent as _ragingbolt_module
from heuristics.alakazam_agent import agent as alakazam_agent
from heuristics.alakazam_v2_agent import agent as alakazam_v2_agent
from heuristics.starmie_agent import agent as starmie_agent

_ragingbolt_module.DEBUG = False  # silence its per-step hand/option dump

# (name, agent fn, deck.csv path). Every package-style agent ships its own
# deck.csv.
HEURISTICS = [
    ("crustle", crustle_agent, "heuristics/crustle_agent/deck.csv"),
    ("abomasnow", abomasnow_agent, "heuristics/abomasnow_agent/deck.csv"),
    ("dragapult", dragapult_agent, "heuristics/dragapult_agent/deck.csv"),
    ("dragapult_v2", dragapult_v2_agent, "heuristics/dragapult_v2_agent/deck.csv"),
    ("iono", iono_agent, "heuristics/iono_agent/deck.csv"),
    ("archaludon", archaludon_agent, "heuristics/archaludon_agent/deck.csv"),
    ("ragingbolt", ragingbolt_agent, "heuristics/ragingbolt_agent/deck.csv"),
    ("alakazam", alakazam_agent, "heuristics/alakazam_agent/deck.csv"),
    ("alakazam_v2", alakazam_v2_agent, "heuristics/alakazam_v2_agent/deck.csv"),
    ("starmie", starmie_agent, "heuristics/starmie_agent/deck.csv"),
]


def _load_deck(deck_path):
    with open(deck_path) as f:
        deck = [int(line) for line in f.readlines() if line.strip()]
    if len(deck) != 60:
        raise AssertionError(f"{deck_path} has {len(deck)} cards, expected 60")
    return deck


def run_heuristic(agent_fn, deck_path):
    """Run one self-play battle. Returns an error string, or None if it passed."""
    deck = _load_deck(deck_path)

    env = make("cabt", configuration={"decks": [deck.copy(), deck.copy()]}, debug=True)
    env.run([agent_fn, agent_fn])

    for i, s in enumerate(env.state):
        if s.get("error"):
            return f"player {i} error: {s['error']}"
        if s["status"] not in ("DONE", "ACTIVE"):
            return f"player {i} status: {s['status']}"
    step0_error = env.steps[0][0].get("error") if env.steps else None
    if step0_error:
        return step0_error
    return None


def main() -> int:
    results = []
    for name, agent_fn, deck_path in HEURISTICS:
        try:
            error = run_heuristic(agent_fn, deck_path)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        results.append((name, error))
        print(f"[{'FAIL' if error else 'PASS'}] {name}" + (f" - {error}" if error else ""))

    failed = [name for name, error in results if error]
    print(f"\n{len(results) - len(failed)}/{len(results)} heuristics passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
