import os

# Point the agent module at the v5 weights before importing it (agent.py loads
# its weights at import time). v5 is the first model trained on the new
# autoregressive Discrete CabtEnv; the old MultiBinary weights (v2-v4) are
# incompatible with the new action head and agent.py will reject them.
os.environ.setdefault("PPO_WEIGHTS", "ppo_starmie_v5_weights.npz")

from kaggle_environments import make
from agent import agent as rl_agent
from heuristics.dragapult_agent import agent as dragapult_agent
import json
from collections import Counter
import hashlib, time


def _load_deck(path):
    with open(path) as f:
        return [int(line) for line in f.readlines() if line.strip()]


# Seat 0 = starmie PPO v4 (deck.csv); seat 1 = dragapult heuristic (its own
# deck). The cabt interpreter takes each player's deck from what their agent
# returns on the first step (obs["select"] is None), so each agent supplies its
# own deck -- the `decks` config below is just for the record, not what drives
# the battle.
deck = _load_deck("deck.csv")
dragapult_deck = _load_deck("heuristics/dragapult_agent/deck.csv")

print("RL (starmie v5) deck size:", len(deck))
print("Counts:", Counter(deck))
print("Deck:", deck)
deck_hash = hashlib.md5(str(deck).encode()).hexdigest()[:8]
print(f"Deck hash: {deck_hash}  Timestamp: {time.time()}")
print("Dragapult deck size:", len(dragapult_deck))

env = make(
    "cabt",
    configuration={"decks": [deck.copy(), dragapult_deck.copy()]},
    debug=True,
)
# Starmie PPO v5 (seat 0) vs the dragapult heuristic (seat 1).
env.run([rl_agent, dragapult_agent])

labels = ["starmie_v5 (RL)", "dragapult (heuristic)"]
for i, agent_state in enumerate(env.state):
    print(
        f"Player {i} [{labels[i]}]: status={agent_state['status']}, "
        f"reward={agent_state.get('reward')}"
    )
    print("error:", agent_state.get("error"))

with open("vis.json", "w") as file:
    json.dump(env.steps[0][0]["visualize"], file)

print("Simulation finished.")
