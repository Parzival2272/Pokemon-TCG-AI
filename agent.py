"""Inference wrapper that exposes the trained ppo_starmie_v2 policy as an
`agent(obs_dict) -> list[int]` callable, matching the other heuristic agent's
interface so it can be dropped into main.py's AGENTS dict.

Runs on plain NumPy weights (ppo_starmie_v2_weights.npz) rather than loading
the sb3_contrib MaskablePPO .zip directly: Kaggle's submission sandbox does
not have torch/sb3_contrib/stable_baselines3 installed (confirmed by a failed
submission -- ModuleNotFoundError: No module named 'sb3_contrib'), so the
agent must not depend on them at runtime. The weights were extracted from
ppo_starmie_v2.zip's policy_net + action_net (see tools/export_policy_weights.py)
and the forward pass below is a hand-written, NumPy-only reimplementation of
MaskableActorCriticPolicy's deterministic action selection -- verified to
produce IDENTICAL selections to the real torch model across 2,120 real
in-game decisions (40 games vs every heuristic opponent) before being wired
in here. If ppo_starmie_v2 is retrained, re-export the weights with that tool;
this module doesn't touch the .zip.

Network shape (MlpPolicy default, extracted from ppo_starmie_v2.zip):
    obs (300,) -> Linear(300,64) -> Tanh -> Linear(64,64) -> Tanh
               -> Linear(64,256) -> reshape (128, 2) per-slot logits
For slot i, "selected" wins over "not selected" iff logits[i,1] > logits[i,0]
(argmax of a 2-way softmax, which softmax's monotonicity makes equivalent to
comparing raw logits directly -- no need to materialize probabilities).
"""

import os

import numpy as np

from training.obs_vectorizer import (
    MAX_OPTIONS,
    VECTOR_SIZE,
    obs_to_vector,
    set_vectorizer_deck,
)

_project_root = os.path.dirname(os.path.abspath(__file__))
_weights_path = os.path.join(_project_root, "ppo_starmie_v2_weights.npz")
_deck_path = os.path.join(_project_root, "deck.csv")

with open(_deck_path) as _f:
    deck: list[int] = [int(line) for line in _f.readlines() if line.strip()]

# The prize-belief block of the observation is computed against this deck
# list; training sets it every episode (CabtEnv.reset), so inference must
# set it too or those 60 features silently degrade to base-rate values.
set_vectorizer_deck(deck)

_w = np.load(_weights_path)
_W0, _B0 = _w["w0"], _w["b0"]
_W2, _B2 = _w["w2"], _w["b2"]
_WA, _BA = _w["wa"], _w["ba"]

# Fail loudly if the weights don't match the current observation layout
# (e.g. obs_vectorizer changed since the weights were exported).
if _W0.shape[1] != VECTOR_SIZE:
    raise RuntimeError(
        f"{os.path.basename(_weights_path)} expects observation size "
        f"{_W0.shape[1]}, but obs_vectorizer produces ({VECTOR_SIZE},). "
        "Re-export the weights (tools/export_policy_weights.py) or check "
        "out the matching obs_vectorizer."
    )
if _WA.shape[0] != MAX_OPTIONS * 2:
    raise RuntimeError(
        f"{os.path.basename(_weights_path)} action head produces "
        f"{_WA.shape[0]} logits, but MAX_OPTIONS*2 = {MAX_OPTIONS * 2}."
    )


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py if used."""
    global deck
    deck = deck_list
    set_vectorizer_deck(deck)


def _predict(obs_dict: dict, n_options: int, min_count: int, max_count: int) -> list[int]:
    obs_vec = obs_to_vector(obs_dict)

    h1 = np.tanh(obs_vec @ _W0.T + _B0)
    h2 = np.tanh(h1 @ _W2.T + _B2)
    logits = (h2 @ _WA.T + _BA).reshape(MAX_OPTIONS, 2)

    # Only the first n_options slots correspond to real options -- mirrors
    # CabtEnv.action_masks(), where "selected" is only ever legal there.
    selected = [i for i in range(n_options) if logits[i, 1] > logits[i, 0]]

    # Same minCount/maxCount clamping CabtEnv.step() applies during training,
    # since the per-slot choice can't jointly enforce a selection count.
    if len(selected) > max_count:
        selected = selected[:max_count]
    if len(selected) < min_count:
        chosen = set(selected)
        for i in range(n_options):
            if len(selected) >= min_count:
                break
            if i not in chosen:
                selected.append(i)
                chosen.add(i)
        selected.sort()

    return selected


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return deck

    options = select.get("option") or []
    n_options = min(len(options), MAX_OPTIONS)
    if n_options == 0:
        return []
    min_count = min(select.get("minCount", 1), n_options)
    max_count = min(select.get("maxCount", 1), n_options)

    # Competition rule: the agent must never crash and must always return a
    # legal action. A bad/malformed observation or an unexpected exception
    # must not forfeit the episode, so fall back to the lowest-index legal
    # options (always in range and within [min_count, max_count]).
    try:
        return _predict(obs_dict, n_options, min_count, max_count)
    except Exception:
        return list(range(min_count))
