"""Run a retired PPO checkpoint as an ordinary heuristic-style opponent.

Once a policy is good enough to be worth training against, freezing it into
heuristics/ turns it into a permanent member of train.py's OPPONENT_POOL: a
fixed skill benchmark that never drifts, unlike the live league snapshots
(training/league.py) which are recreated from scratch on every run.

Two things have to be frozen together for that to keep working:

  * the weights (.npz, from tools/export_policy_weights.py), and
  * the observation encoder they were trained against.

The second one is the reason this module takes a `vectorizer` module rather
than importing training.obs_vectorizer directly. obs_vectorizer changes
whenever the feature set is extended, and any such change silently invalidates
every older export (agent.py has an explicit VECTOR_SIZE guard for exactly
this). A frozen opponent that imported the live encoder would therefore break
the moment you improved the observation -- i.e. precisely when you most want a
stable baseline to measure the improvement against. So each frozen agent
package vendors a verbatim snapshot of the obs_vectorizer it was trained with
and passes it in here.

A vendored encoder also fixes a subtlety the live league only gets away with
by playing mirror matches: obs_vectorizer keeps the deck used for its
prize/deck-count belief features in module globals, and CabtEnv.reset() sets
those to the *learner's* deck. The vendored copy has its own globals, set once
to this agent's own deck, so a frozen opponent reasons about its own decklist
even when the learner is on a different one.

Inference is NumPy-only (no torch / sb3_contrib), matching agent.py -- so
these agents also work inside the Kaggle submission sandbox.
"""

import os

import numpy as np


class FrozenPolicyAgent:
    """`agent(obs_dict) -> list[int]` over an exported MaskablePPO policy.

    Network shape (MlpPolicy, see training/cabt_env.POLICY_NET_ARCH):
        obs -> Linear -> Tanh -> Linear -> Tanh -> Linear -> logits
    The action space is autoregressive Discrete(MAX_OPTIONS + 1): one option
    is chosen per forward pass, with the partial selection fed back in via
    obs_to_vector(picked=...), until STOP (index MAX_OPTIONS) is chosen or
    maxCount options have been picked. This mirrors CabtEnv.step /
    CabtEnv.action_masks, which is how the policy was trained.

    Args:
        weights_path: .npz written by tools/export_policy_weights.py.
        deck: 60-card ID list this policy was trained on. Installed into
            `vectorizer`'s module globals, which is why the vectorizer must
            be a private snapshot and not the shared training one.
        vectorizer: module exposing MAX_OPTIONS / VECTOR_SIZE /
            obs_to_vector / set_vectorizer_deck.
        temperature: 0.0 (default) picks the argmax, reproducing the exact
            play the checkpoint would make as a submission. Above 0 samples
            from softmax(logits / temperature), which makes the opponent
            harder to overfit to a single exploitable line -- useful when the
            learner's win rate against it pins at 100%.
        seed: RNG seed, used only when temperature > 0.
    """

    def __init__(self, weights_path, deck, vectorizer, temperature=0.0, seed=0):
        self._vec = vectorizer
        self._temperature = float(temperature)
        self._rng = np.random.default_rng(seed)

        w = np.load(weights_path)
        self._W0, self._B0 = w["w0"], w["b0"]
        self._W2, self._B2 = w["w2"], w["b2"]
        self._WA, self._BA = w["wa"], w["ba"]

        # The vendored vectorizer should make these impossible, but a wrong
        # .npz/obs_vectorizer pairing in a new package would otherwise show up
        # as quietly garbage play rather than an error.
        if self._W0.shape[1] != vectorizer.VECTOR_SIZE:
            raise RuntimeError(
                f"{os.path.basename(weights_path)} expects observation size "
                f"{self._W0.shape[1]}, but its vendored obs_vectorizer "
                f"produces {vectorizer.VECTOR_SIZE}."
            )
        if self._WA.shape[0] != vectorizer.MAX_OPTIONS + 1:
            raise RuntimeError(
                f"{os.path.basename(weights_path)} has "
                f"{self._WA.shape[0]} action logits, but its vendored "
                f"obs_vectorizer has MAX_OPTIONS + 1 = "
                f"{vectorizer.MAX_OPTIONS + 1}."
            )

        self.set_deck(deck)

    def set_deck(self, deck):
        """Point the prize/deck-count belief features at `deck`.

        Only affects this agent: `self._vec` is its own module object.
        """
        self.deck = list(deck)
        self._vec.set_vectorizer_deck(self.deck)

    def _logits(self, obs_dict, picked):
        h = np.tanh(self._vec.obs_to_vector(obs_dict, picked=picked) @ self._W0.T + self._B0)
        h = np.tanh(h @ self._W2.T + self._B2)
        return h @ self._WA.T + self._BA

    def _choose(self, logits, legal):
        masked = np.where(legal, logits, -np.inf)
        if self._temperature <= 0:
            return int(np.argmax(masked))
        masked = masked / self._temperature
        p = np.exp(masked - masked.max())
        p /= p.sum()
        return int(self._rng.choice(len(p), p=p))

    def _predict(self, obs_dict, n_options, min_count, max_count):
        stop = self._vec.MAX_OPTIONS
        picked: list[int] = []
        while True:
            legal = np.zeros(stop + 1, dtype=bool)
            if len(picked) < max_count:
                legal[:n_options] = True
                legal[picked] = False
            if len(picked) >= min_count:
                legal[stop] = True

            action = self._choose(self._logits(obs_dict, picked), legal)
            if action == stop:
                break
            picked.append(action)
            if len(picked) >= max_count:
                break

        picked.sort()
        return picked

    def __call__(self, obs_dict):
        select = obs_dict.get("select")
        if select is None:  # deck-submission phase
            return self.deck

        options = select.get("option") or []
        n_options = min(len(options), self._vec.MAX_OPTIONS)
        if n_options == 0:
            return []
        max_count = min(select.get("maxCount", 1), n_options)
        min_count = min(select.get("minCount", 1), max_count)

        # Same contract as every other agent here: never crash, always return
        # a legal selection. As a training opponent a raised exception would
        # be swallowed by CabtEnv anyway, but it would silently turn the
        # episode into random play.
        try:
            return self._predict(obs_dict, n_options, min_count, max_count)
        except Exception:
            return list(range(min_count))


def load_deck(path):
    """Read a 60-card deck file (one card ID per line)."""
    with open(path) as f:
        deck = [int(line) for line in f if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} must contain 60 cards, got {len(deck)}")
    return deck
