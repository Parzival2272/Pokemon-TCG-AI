import os

from crustle_agent import agent as crustle_agent
from crustle_agent import set_deck as crustle_set_deck
from abomasnow_agent import agent as abomasnow_agent
from abomasnow_agent import set_deck as abomasnow_set_deck
from dragapult_agent_agent import agent as dragapult_agent_agent
from dragapult_agent_agent import set_deck as dragapult_agent_set_deck
from dragapult_agent import agent as dragapult_agent
from dragapult_agent import set_deck as dragapult_set_deck
from iono_agent import agent as iono_agent
from iono_agent import set_deck as iono_set_deck
from archaludon_agent import agent as archaludon_agent
from archaludon_agent import set_deck as archaludon_set_deck
from dragapult_v2_agent import agent as dragapult_v2_agent
from dragapult_v2_agent import set_deck as dragapult_v2_set_deck
from ragingbolt_agent import agent as ragingbolt_agent
from ragingbolt_agent import set_deck as ragingbolt_set_deck

_KAGGLE_DECK_PATH = "/kaggle_simulations/agent/deck.csv"
if os.path.exists(_KAGGLE_DECK_PATH):
    _deck_path = _KAGGLE_DECK_PATH
elif "__file__" in globals():
    _deck_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
else:
    _deck_path = "deck.csv"

with open(_deck_path) as _f:
    _deck = [int(line) for line in _f.readlines() if line.strip()]

AGENTS = {
    "crustle": crustle_agent,
    "abomasnow": abomasnow_agent,
    "dragapult_agent": dragapult_agent_agent,
    "dragapult": dragapult_agent,
    "iono": iono_agent,
    "archaludon": archaludon_agent,
    "dragapult_v2": dragapult_v2_agent,
    "ragingbolt": ragingbolt_agent,
}

SET_DECKS = {
    "crustle": crustle_set_deck,
    "abomasnow": abomasnow_set_deck,
    "dragapult_agent": dragapult_agent_set_deck,
    "dragapult": dragapult_set_deck,
    "iono": iono_set_deck,
    "archaludon": archaludon_set_deck,
    "dragapult_v2": dragapult_v2_set_deck,
    "ragingbolt": ragingbolt_set_deck,
}

ACTIVE_AGENT = "crustle"

# deck.csv (next to main.py) is the active agent's deck: when switching
# ACTIVE_AGENT, copy that agent's <name>_agent/deck.csv over deck.csv.
SET_DECKS[ACTIVE_AGENT](_deck)


def agent(obs_dict: dict) -> list:
    return AGENTS[ACTIVE_AGENT](obs_dict)
