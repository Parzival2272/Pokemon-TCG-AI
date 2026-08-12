"""Trained PPO checkpoints as fixed CabtEnv opponents.

training/league.py already runs *this* run's own frozen snapshots as opponents;
this module is the other case -- a finished model from a PREVIOUS run, pinned
into OPPONENT_POOL alongside the heuristics so the learner has a strong,
non-heuristic baseline to beat (currently ppo_starmie_v16 on the starmie list).

Two things make it more than a thin wrapper around league._SnapshotAgent:

1. It pilots its OWN deck, not the learner's. Snapshot opponents mirror-match
   because that's the deck they were trained on; a cross-deck PPO opponent is
   the same situation as a heuristic opponent, so it needs obs_vectorizer's
   global deck (which CabtEnv.reset points at the LEARNER's list, since prize
   belief is a learner-perspective feature) swapped to its own list for the
   duration of its decision -- otherwise its 60 deck-slot flags and 30
   unseen-copies counts are read off a decklist it isn't playing. __call__
   swaps and restores around each decision; one battle per process and no
   threads, so the global is safe to borrow.

2. It runs on an exported .npz (tools/export_policy_weights.py) with a NumPy
   forward pass, not a torch policy. Same reasoning as agent.py, plus: each
   SubprocVecEnv worker would otherwise hold its own MaskablePPO, and the
   weights are the only part of it this needs.

Observation-layout drift is handled explicitly. A checkpoint's first layer
pins the VECTOR_SIZE it was trained on, and obs_vectorizer has grown since:
commit 4b4ec78 appended NEXT_STATE_FEATURES to the end of each per-option
slice, taking the vector from 3312 to 5616 while leaving the prefix and the
first 23 features of every option untouched. So an older policy's input is an
exact sub-slice of today's vector, and _project cuts it (see _opt_width).
That works for a purely *appended* per-option block -- if a future change
inserts features mid-vector or widens the prefix, this projection is silently
wrong and the checkpoint must be retrained/re-exported instead.
"""

import os

import numpy as np

from training import obs_vectorizer as ov

# Mirrors CabtEnv.STOP_ACTION, as league.py does -- same reason (no env import
# in every worker for one constant).
STOP_ACTION = ov.MAX_OPTIONS

# Width of everything before the per-option block (board, hand, deck belief,
# state/log scalars, opponent semantics). Layout-drift handling below assumes
# this part is unchanged across checkpoint generations.
_PREFIX = ov.VECTOR_SIZE - ov.OPTION_FEATURES * ov.MAX_OPTIONS


def _resolve_weights(path: str) -> str:
    """Locate a weights export: a bare filename is looked up in models/ (where
    tools/export_policy_weights.py writes), a path with a directory component
    is taken as-is. Same rule as agent.py's resolver."""
    if os.path.isabs(path) or os.path.dirname(path):
        return path
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", path
    )


class PPOOpponent:
    """CabtEnv-style opponent -- callable(obs_dict) -> list[int] -- piloting a
    trained policy loaded from an .npz weights export.

    Args:
        weights: .npz from tools/export_policy_weights.py (bare filename is
            resolved against models/).
        deck: the 60-card list this opponent pilots. MUST be the same list the
            pool entry carries, or the deck-belief features it sees describe a
            different deck than the one the engine deals it.
        name: label used in load errors only; WinRateCallback buckets on the
            pool entry's name, not this.
        deterministic: greedy argmax over legal logits (default), matching how
            the checkpoint plays at inference in agent.py -- i.e. the opponent
            plays at the strength it was measured at. Set False to sample from
            the masked distribution instead, which trades some of that strength
            for line variety (what league.py's snapshots do, since they are
            near-copies of the learner and would otherwise be memorizable).
    """

    def __init__(self, weights, deck, name=None, deterministic=True):
        self.weights_path = _resolve_weights(weights)
        self.deck = list(deck)
        self.name = name or os.path.basename(self.weights_path)
        self.deterministic = deterministic
        # Loaded lazily on first decision: OPPONENT_POOL is built at train.py
        # import time in the parent process, and every SubprocVecEnv worker
        # re-imports it anyway, so eager loading would just be a per-process
        # cost paid by workers that may never draw this opponent.
        self._w = None
        self._opt_width = 0

    def _load(self):
        w = np.load(self.weights_path)
        self._w = (w["w0"], w["b0"], w["w2"], w["b2"], w["wa"], w["ba"])
        expected = int(w["w0"].shape[1])

        # How many of today's OPTION_FEATURES this checkpoint was trained on
        # (see the module docstring). Must divide evenly and not exceed the
        # current width -- anything else is a layout change this projection
        # cannot express.
        opt_span = expected - _PREFIX
        width, rem = divmod(opt_span, ov.MAX_OPTIONS)
        if opt_span < 0 or rem or not 0 < width <= ov.OPTION_FEATURES:
            raise RuntimeError(
                f"{self.name} expects an observation of {expected} floats, "
                f"which is not a prefix-compatible slice of the current "
                f"{ov.VECTOR_SIZE} ({_PREFIX} + {ov.OPTION_FEATURES} x "
                f"{ov.MAX_OPTIONS}). obs_vectorizer changed in a way that is "
                "not a per-option append; retrain or re-export this policy."
            )
        self._opt_width = width

        if int(w["wa"].shape[0]) != ov.MAX_OPTIONS + 1:
            raise RuntimeError(
                f"{self.name} action head produces {w['wa'].shape[0]} logits, "
                f"but MAX_OPTIONS + 1 = {ov.MAX_OPTIONS + 1}."
            )

    def _project(self, vec):
        """Cut today's observation down to the slice this checkpoint expects."""
        if self._opt_width == ov.OPTION_FEATURES:
            return vec
        opts = vec[_PREFIX:].reshape(ov.MAX_OPTIONS, ov.OPTION_FEATURES)
        return np.concatenate([vec[:_PREFIX], opts[:, : self._opt_width].ravel()])

    def _logits(self, obs_dict, picked):
        w0, b0, w2, b2, wa, ba = self._w
        vec = self._project(ov.obs_to_vector(obs_dict, picked=picked))
        h1 = np.tanh(vec @ w0.T + b0)
        h2 = np.tanh(h1 @ w2.T + b2)
        return h2 @ wa.T + ba

    def __call__(self, obs_dict):
        if self._w is None:
            self._load()

        select = obs_dict.get("select") or {}
        n = min(len(select.get("option") or []), ov.MAX_OPTIONS)
        if n == 0:
            return []
        max_count = min(select.get("maxCount", 1), n)
        min_count = min(select.get("minCount", 1), max_count)

        # The lookahead block is the part of the vector this policy doesn't
        # consume, and building it runs a determinized ptcg.api search per
        # option -- by far the most expensive thing in obs_to_vector. Hiding
        # search_begin_input makes _next_state_block_for_option fall back to
        # the (discarded) current-state block without searching.
        if obs_dict.get("search_begin_input") is not None:
            obs_dict = {k: v for k, v in obs_dict.items() if k != "search_begin_input"}

        prev_deck = list(ov.VECTORIZER_DECK)
        ov.set_vectorizer_deck(self.deck)  # see module docstring, point 1
        try:
            picked: list[int] = []
            while len(picked) < max_count:
                legal = np.zeros(ov.MAX_OPTIONS + 1, dtype=bool)
                legal[:n] = True
                legal[picked] = False
                if len(picked) >= min_count:
                    legal[STOP_ACTION] = True

                logits = np.where(legal, self._logits(obs_dict, picked), -np.inf)
                if self.deterministic:
                    action = int(np.argmax(logits))
                else:
                    p = np.exp(logits - logits.max())
                    action = int(np.random.choice(len(p), p=p / p.sum()))
                if action == STOP_ACTION:
                    break
                picked.append(action)
        finally:
            ov.set_vectorizer_deck(prev_deck)

        picked.sort()
        return picked