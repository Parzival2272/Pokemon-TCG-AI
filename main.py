import os
from crustle_agent import agent as crustle_agent

_KAGGLE_DECK_PATH = "/kaggle_simulations/agent/deck.csv"
if os.path.exists(_KAGGLE_DECK_PATH):
    _deck_path = _KAGGLE_DECK_PATH
elif "__file__" in globals():
    _deck_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
else:
    _deck_path = "deck.csv"

with open(_deck_path) as _f:
    _deck = [int(line) for line in _f.readlines() if line.strip()]

from crustle_agent import set_deck as _set_deck

_set_deck(_deck)

AGENTS = {
    "crustle": crustle_agent,
    # "other": other_agents,
}

ACTIVE_AGENT = "crustle"


def agent(obs_dict: dict) -> list:
    return AGENTS[ACTIVE_AGENT](obs_dict)
