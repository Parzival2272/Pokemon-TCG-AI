import os

from heuristics.crustle_agent import agent as crustle_agent
from agent import agent as rl_agent
from agent import set_deck as rl_set_deck
from heuristics.crustle_agent import set_deck as crustle_set_deck
from heuristics.abomasnow_agent import agent as abomasnow_agent
from heuristics.abomasnow_agent import set_deck as abomasnow_set_deck
from heuristics.dragapult_agent import agent as dragapult_agent
from heuristics.dragapult_agent import set_deck as dragapult_set_deck
from heuristics.iono_agent import agent as iono_agent
from heuristics.iono_agent import set_deck as iono_set_deck
from heuristics.archaludon_agent import agent as archaludon_agent
from heuristics.archaludon_agent import set_deck as archaludon_set_deck
from heuristics.dragapult_v2_agent import agent as dragapult_v2_agent
from heuristics.dragapult_v2_agent import set_deck as dragapult_v2_set_deck
from heuristics.ragingbolt_agent import agent as ragingbolt_agent
from heuristics.ragingbolt_agent import set_deck as ragingbolt_set_deck
from heuristics.alakazam_agent import agent as alakazam_agent
from heuristics.alakazam_agent import set_deck as alakazam_set_deck
from heuristics.alakazam_v2_agent import agent as alakazam_v2_agent
from heuristics.alakazam_v2_agent import set_deck as alakazam_v2_set_deck
from heuristics.starmie_agent import agent as starmie_agent
from heuristics.starmie_agent import set_deck as starmie_set_deck

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
    "ppo_crustle": rl_agent,
    # "other": other_agents,
    "abomasnow": abomasnow_agent,
    "dragapult": dragapult_agent,
    "iono": iono_agent,
    "archaludon": archaludon_agent,
    "dragapult_v2": dragapult_v2_agent,
    "ragingbolt": ragingbolt_agent,
    "alakazam": alakazam_agent,
    "alakazam_v2": alakazam_v2_agent,
    "starmie": starmie_agent,
}

SET_DECKS = {
    "crustle": crustle_set_deck,
    "ppo_crustle": rl_set_deck,
    "abomasnow": abomasnow_set_deck,
    "dragapult": dragapult_set_deck,
    "iono": iono_set_deck,
    "archaludon": archaludon_set_deck,
    "dragapult_v2": dragapult_v2_set_deck,
    "ragingbolt": ragingbolt_set_deck,
    "alakazam": alakazam_set_deck,
    "alakazam_v2": alakazam_v2_set_deck,
    "starmie": starmie_set_deck,
}

ACTIVE_AGENT = "ppo_crustle"

# deck.csv (next to main.py) is the active agent's deck: when switching
# ACTIVE_AGENT, copy that agent's heuristics/<name>_agent/deck.csv over deck.csv.
SET_DECKS[ACTIVE_AGENT](_deck)


def agent(obs_dict: dict) -> list:
    return AGENTS[ACTIVE_AGENT](obs_dict)
