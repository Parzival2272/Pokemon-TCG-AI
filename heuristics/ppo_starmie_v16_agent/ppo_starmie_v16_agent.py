"""The v16 starmie PPO checkpoint, frozen as a heuristic-style opponent.

This is the model agent.py shipped as the Kaggle submission (models/
ppo_starmie_v16_weights.npz). Freezing it here makes it a permanent entry in
train.py's OPPONENT_POOL, so every future run is graded against "the last
version we shipped" -- a fixed bar, unlike the live league (training/
league.py), whose snapshots are rebuilt from the current learner each run and
so improve alongside it.

Everything this agent needs is vendored in this package and must not be
repointed at the live training/ modules:

    weights.npz        copy of models/ppo_starmie_v16_weights.npz
    obs_vectorizer.py  verbatim snapshot of training/obs_vectorizer.py as of
                       the v16 training run (VECTOR_SIZE 3312, MAX_OPTIONS
                       128, POLICY_NET_ARCH [256, 256])
    deck.csv           copy of the root deck.csv v16 was trained on

See heuristics/frozen_ppo.py for why the encoder is copied rather than
imported. To retire a newer checkpoint the same way, copy this directory,
swap the three data files, and add the entry to train.py.
"""

import os

from heuristics.frozen_ppo import FrozenPolicyAgent, load_deck

from . import obs_vectorizer

_HERE = os.path.dirname(os.path.abspath(__file__))

# Greedy (temperature=0) so this opponent plays exactly the lines the v16
# submission would: "beat this agent" then means "beat what we shipped".
# Raise the temperature if a learner pins at ~100% win rate against it and
# you want variety instead of a single exploitable script.
_agent = FrozenPolicyAgent(
    weights_path=os.path.join(_HERE, "weights.npz"),
    deck=load_deck(os.path.join(_HERE, "deck.csv")),
    vectorizer=obs_vectorizer,
    temperature=float(os.environ.get("PPO_V16_TEMPERATURE", 0.0)),
)

deck = _agent.deck


def set_deck(deck_list):
    """Override the packaged deck (matches the other heuristics' interface).

    Rarely wanted here: the policy was trained on deck.csv, and it reasons
    about its own decklist through the vendored vectorizer's belief features.
    """
    global deck
    _agent.set_deck(deck_list)
    deck = _agent.deck


def agent(obs_dict):
    return _agent(obs_dict)
