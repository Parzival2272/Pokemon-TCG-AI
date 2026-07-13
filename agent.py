"""Inference wrapper that exposes the trained ppo_crustle.zip model as an
`agent(obs_dict) -> list[int]` callable, matching crustle_agent's interface
so it can be dropped into main.py's AGENTS dict.

ppo_crustle.zip is a MaskablePPO model (see training/train.py) trained on
vectorized observations (training/obs_vectorizer.obs_to_vector) with a
MultiBinary(MAX_OPTIONS) action space. This module does the same
obs_dict -> vector conversion and action-mask construction that
training/cabt_env.py does during training, then converts the model's
0/1 action vector back into the list of selected option indices the
native engine expects.
"""

import os

import numpy as np
from sb3_contrib import MaskablePPO

from training.obs_vectorizer import MAX_OPTIONS, obs_to_vector

_project_root = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_project_root, "ppo_crustle.zip")
_deck_path = os.path.join(_project_root, "deck.csv")

with open(_deck_path) as _f:
    deck: list[int] = [int(line) for line in _f.readlines() if line.strip()]

_model = MaskablePPO.load(_model_path)


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py if used."""
    global deck
    deck = deck_list


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

    obs_vec = obs_to_vector(obs_dict)

    # Mirrors CabtEnv.action_masks(): column 0 is "not selected", column 1
    # is "selected"; only the first n_options slots are legal to pick.
    mask = np.zeros((MAX_OPTIONS, 2), dtype=bool)
    mask[:, 0] = True
    mask[:n_options, 1] = True

    action, _ = _model.predict(obs_vec, action_masks=mask, deterministic=True)
    selected = [i for i in range(n_options) if action[i]]

    # Same minCount/maxCount clamping CabtEnv.step() applies during training,
    # since the mask can't jointly enforce a selection count.
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
