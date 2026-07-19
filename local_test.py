from kaggle_environments import make
from agent import agent as rl_agent
import json
from collections import Counter
import hashlib, time

with open("deck.csv") as f:
    deck = [int(line) for line in f.readlines() if line.strip()]

print("Deck size:", len(deck))
print("Counts:", Counter(deck))
print("Deck:", deck)
deck_hash = hashlib.md5(str(deck).encode()).hexdigest()[:8]
print(f"Deck hash: {deck_hash}  Timestamp: {time.time()}")

env = make("cabt", configuration={"decks": [deck.copy(), deck.copy()]}, debug=True)
# Starmie PPO vs itself (both seats run the same model; deck.csv = starmie deck).
env.run([rl_agent, rl_agent])

for i, agent_state in enumerate(env.state):
    print(
        f"Player {i}: status={agent_state['status']}, reward={agent_state.get('reward')}"
    )
    print("error:", agent_state.get("error"))

with open("vis.json", "w") as file:
    json.dump(env.steps[0][0]["visualize"], file)

print("Simulation finished.")
