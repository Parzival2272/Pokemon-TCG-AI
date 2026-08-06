"""Inference wrapper that exposes the trained MaskablePPO policy as an
`agent(obs_dict) -> list[int]` callable, matching the heuristic agents'
interface so it can be dropped into main.py's AGENTS dict.

The shipped default is the Mega Kangaskhan ex / Crustle policy
(models/ppo_crustle_v2_weights.npz), trained on crustle_deck.csv -- which is
what the root deck.csv must be a copy of, since the observation's prize-belief
block is computed against the deck loaded below.

Runs on plain NumPy weights (an .npz exported by tools/export_policy_weights.py)
rather than loading the sb3_contrib MaskablePPO .zip directly: Kaggle's
submission sandbox has no torch/sb3_contrib/stable_baselines3 (confirmed by a
failed submission -- ModuleNotFoundError: No module named 'sb3_contrib'), so
this module must not import them at runtime. The forward pass below is a
NumPy-only reimplementation of the policy's deterministic (greedy) action
selection. Point it at a different export via the PPO_WEIGHTS env var, and
re-export after every retrain so the submission bundle matches the model.

The action space is autoregressive Discrete(MAX_OPTIONS + 1) (see
training/cabt_env.py): the policy picks ONE option per forward pass, or the STOP
action (index MAX_OPTIONS) to finish, and a decision needing k picks is a
sequence of k greedy passes. A single engine `select` wants the whole subset at
once, so this wrapper produces it by looping the network internally -- feeding
each pass the partial selection via obs_to_vector(obs_dict, picked=...) -- the
same rollout CabtEnv performs across gym steps during training.

Network shape (POLICY_NET_ARCH in training/cabt_env.py, currently [256, 256]);
the layer widths are read off the loaded weights, so only the depth is fixed:
    obs (VECTOR_SIZE,) -> Linear(.,H) -> Tanh -> Linear(H,H) -> Tanh
                       -> Linear(H, MAX_OPTIONS + 1) logits
The chosen action is argmax over the currently-legal logits (masking is
monotone-safe, so comparing raw logits matches comparing softmax probabilities).
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


def _resolve_weights(path: str) -> str:
    """Locate a weights export. A bare filename is looked up in models/ (where
    training and tools/export_policy_weights.py write) and then next to this
    file, so both the repo layout and an older flat submission bundle work. A
    path with a directory component is taken as-is."""
    if os.path.isabs(path) or os.path.dirname(path):
        return path
    candidates = [
        os.path.join(_project_root, "models", path),
        os.path.join(_project_root, path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]  # missing: report the models/ path in the load error


# Kaggle submission always runs the crustle v2 weights (the default). Local
# tooling (e.g. local_test.py running an older export against itself) can point
# this at a different file via the PPO_WEIGHTS env var without touching the
# submission default.
_weights_path = _resolve_weights(
    os.environ.get("PPO_WEIGHTS", "ppo_crustle_v2_weights.npz")
)
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
if _WA.shape[0] != MAX_OPTIONS + 1:
    raise RuntimeError(
        f"{os.path.basename(_weights_path)} action head produces "
        f"{_WA.shape[0]} logits, but MAX_OPTIONS + 1 = {MAX_OPTIONS + 1}. "
        "This weights file is from the old MultiBinary policy; retrain on the "
        "autoregressive Discrete CabtEnv and re-export."
    )


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py if used."""
    global deck
    deck = deck_list
    set_vectorizer_deck(deck)


def _predict(obs_dict: dict, n_options: int, min_count: int, max_count: int) -> list[int]:
    """Greedily roll out the autoregressive policy for one engine `select`,
    returning the chosen option indices. Mirrors CabtEnv.step /
    CabtEnv.action_masks: pick one legal option per pass, feeding the partial
    selection back in, until STOP is chosen or maxCount is reached."""
    picked: list[int] = []
    while True:
        obs_vec = obs_to_vector(obs_dict, picked=picked)
        h1 = np.tanh(obs_vec @ _W0.T + _B0)
        h2 = np.tanh(h1 @ _W2.T + _B2)
        logits = h2 @ _WA.T + _BA  # (MAX_OPTIONS + 1,)

        # Legal-action mask, identical to CabtEnv.action_masks(): unpicked real
        # options while the selection isn't full, plus STOP once minCount met.
        legal = np.zeros(MAX_OPTIONS + 1, dtype=bool)
        if len(picked) < max_count:
            for i in range(n_options):
                if i not in picked:
                    legal[i] = True
        if len(picked) >= min_count:
            legal[MAX_OPTIONS] = True

        action = int(np.argmax(np.where(legal, logits, -np.inf)))
        if action == MAX_OPTIONS:  # STOP
            break
        picked.append(action)
        if len(picked) >= max_count:
            break

    picked.sort()
    return picked


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")
    if select is None:
        return deck

    options = select.get("option") or []
    n_options = min(len(options), MAX_OPTIONS)
    if n_options == 0:
        return []
    max_count = min(select.get("maxCount", 1), n_options)
    min_count = min(select.get("minCount", 1), max_count)

    # Competition rule: the agent must never crash and must always return a
    # legal action. A bad/malformed observation or an unexpected exception
    # must not forfeit the episode, so fall back to the lowest-index legal
    # options (always in range and within [min_count, max_count]).
    try:
        return _predict(obs_dict, n_options, min_count, max_count)
    except Exception:
        return list(range(min_count))
