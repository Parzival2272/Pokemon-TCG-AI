import os

# Point the agent module at the v3 weights before importing it (agent.py loads
# its weights at import time). Kaggle still defaults to v2; this override only
# affects this local test.
os.environ.setdefault("PPO_WEIGHTS", "ppo_starmie_v3_weights.npz")

from kaggle_environments import make
from agent import agent as rl_agent
from heuristics.starmie_agent import agent as starmie_heuristic
import json
from collections import Counter
import hashlib, time


def _load_deck(path):
    with open(path) as f:
        return [int(line) for line in f.readlines() if line.strip()]


# Seat 0 = starmie PPO v3 (deck.csv); seat 1 = starmie heuristic. The cabt
# interpreter takes each player's deck from what their agent returns on the
# first step (obs["select"] is None), so each agent supplies its own deck.
# starmie_agent auto-loads its own deck.csv (a different starmie variant than
# the root deck.csv the RL agent plays), so no set_deck override is needed.
deck = _load_deck("deck.csv")
starmie_h_deck = _load_deck("heuristics/starmie_agent/deck.csv")

print("RL (starmie v3) deck size:", len(deck))
print("Counts:", Counter(deck))
print("Deck:", deck)
deck_hash = hashlib.md5(str(deck).encode()).hexdigest()[:8]
print(f"Deck hash: {deck_hash}  Timestamp: {time.time()}")
print("Starmie heuristic deck size:", len(starmie_h_deck))

env = make(
    "cabt",
    configuration={"decks": [deck.copy(), starmie_h_deck.copy()]},
    debug=True,
)
# Starmie PPO v3 (seat 0) vs the starmie heuristic (seat 1).
env.run([rl_agent, starmie_heuristic])

labels = ["starmie_v3 (RL)", "starmie (heuristic)"]
for i, agent_state in enumerate(env.state):
    print(
        f"Player {i} [{labels[i]}]: status={agent_state['status']}, "
        f"reward={agent_state.get('reward')}"
    )
    print("error:", agent_state.get("error"))

with open("vis.json", "w") as file:
    json.dump(env.steps[0][0]["visualize"], file)

print("Simulation finished.")
