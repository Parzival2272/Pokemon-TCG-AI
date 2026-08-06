"""Local match: the iono heuristic vs the crustle heuristic.

Each side plays the deck.csv inside its own package
(heuristics/iono_agent/deck.csv and heuristics/crustle_agent/deck.csv).

The cabt interpreter takes each player's deck from what their agent returns on
the first step (obs["select"] is None), so each agent supplies its own deck and
the `decks` config below is only for the record.

The engine shuffles internally with no exposed seed, so every game differs.
Seats are alternated across games (the going-first advantage is real) and the
summary is a win rate over GAMES games.

Usage:
    python local_test.py [games]
"""

import json
import sys
from collections import Counter

from kaggle_environments import make

from heuristics.iono_agent import agent as iono_agent
from heuristics.crustle_agent import agent as crustle_agent

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

A_LABEL = "iono (heuristic)"
B_LABEL = "crustle (heuristic)"


def _load_deck(path):
    with open(path) as f:
        deck = [int(line) for line in f.readlines() if line.strip()]
    if len(deck) != 60:
        raise AssertionError(f"{path} has {len(deck)} cards, expected 60")
    return deck


a_deck = _load_deck("heuristics/iono_agent/deck.csv")
b_deck = _load_deck("heuristics/crustle_agent/deck.csv")

print(f"{A_LABEL} deck: 60 cards, {len(set(a_deck))} unique")
print("Counts:", Counter(a_deck))
print(f"{B_LABEL} deck: 60 cards, {len(set(b_deck))} unique")
print(f"Running {GAMES} games, alternating seats...\n")


def play(a_seat: int):
    """Run one battle with the iono agent in `a_seat`. Returns
    (result, error) where result is 'win' | 'loss' | 'draw' from iono's side."""
    decks = [None, None]
    agents = [None, None]
    decks[a_seat], agents[a_seat] = a_deck.copy(), iono_agent
    decks[1 - a_seat], agents[1 - a_seat] = b_deck.copy(), crustle_agent

    env = make("cabt", configuration={"decks": decks}, debug=True)
    env.run(agents)

    for i, s in enumerate(env.state):
        if s.get("error") or s["status"] not in ("DONE", "ACTIVE"):
            who = A_LABEL if i == a_seat else B_LABEL
            return None, f"player {i} [{who}] status={s['status']} error={s.get('error')}"
    step0_error = env.steps[0][0].get("error") if env.steps else None
    if step0_error:
        return None, step0_error

    reward = env.state[a_seat].get("reward")
    result = {1: "win", -1: "loss", 0: "draw"}.get(reward, f"reward={reward}")
    return (result, env), None


tally = Counter()
errors = []
last_env = None

for game in range(GAMES):
    a_seat = game % 2
    outcome, error = play(a_seat)
    if error:
        errors.append(f"game {game + 1}: {error}")
        print(f"game {game + 1:>3}  iono seat {a_seat}  ERROR  {error}")
        continue
    result, last_env = outcome
    tally[result] += 1
    tally[f"seat{a_seat}_{result}"] += 1
    print(f"game {game + 1:>3}  iono seat {a_seat}  -> iono {result}")

played = tally["win"] + tally["loss"] + tally["draw"]
print(f"\n=== {A_LABEL} vs {B_LABEL} over {played} completed games ===")
if played:
    print(f"iono wins:   {tally['win']}  ({tally['win'] / played:.1%})")
    print(f"iono losses: {tally['loss']}")
    print(f"Draws:       {tally['draw']}")
    print(
        f"  as seat 0 (first): {tally['seat0_win']}W-{tally['seat0_loss']}L-{tally['seat0_draw']}D"
    )
    print(
        f"  as seat 1:         {tally['seat1_win']}W-{tally['seat1_loss']}L-{tally['seat1_draw']}D"
    )
if errors:
    print(f"\n{len(errors)} game(s) failed:")
    for e in errors:
        print("  " + e)

# Replay of the last completed game, for visualizer.html.
if last_env is not None:
    with open("vis.json", "w") as file:
        json.dump(last_env.steps[0][0]["visualize"], file)

sys.exit(1 if errors else 0)
