import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from training.obs_vectorizer import (
    MAX_OPTIONS,
    VECTOR_SIZE,
    obs_to_vector,
    set_vectorizer_deck,
)
import numpy as np
import gymnasium as gym

DECK_PATH = "deck.csv"


def _load_deck():
    """Load a 60-card deck (one card ID per line) from DECK_PATH."""
    with open(DECK_PATH, "r") as f:
        deck = [int(x) for x in f.read().splitlines() if x.strip()]
    if len(deck) != 60:
        raise ValueError(
            f"A deck must contain exactly 60 cards, but got {len(deck)} cards."
        )
    return deck


class CabtEnv(gym.Env):
    """Wraps a CABT game as a Gymnasium environment."""

    OBS_SIZE = VECTOR_SIZE
    MAX_OPTIONS = MAX_OPTIONS

    def __init__(self, opponent_agent=None):
        """
        Args:
            opponent_agent: Optional callable(obs_dict) -> list[int], e.g.
                `crustle_agent.agent`. When set, the learner plays only one
                side (randomized each episode) and this callable makes every
                decision for the other side. When None (default), both
                sides are controlled by the RL policy (pure self-play).
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.OBS_SIZE,), dtype=np.float32
        )
        # A subset of the MAX_OPTIONS slots is selected per step, since some
        # decisions require choosing 0, 2, or any other count of options at
        # once (see SelectData.minCount/maxCount), not just exactly 1.
        self.action_space = gym.spaces.MultiBinary(self.MAX_OPTIONS)

        self._opponent_agent = opponent_agent
        self._learner_index = 0
        self._obs = None
        self._n_options = 0
        self._min_count = 0
        self._max_count = 0
        self._battle = None

    def _play_opponent_until_learner_turn(self, obs_dict):
        """Auto-play the non-learner side with `opponent_agent` until it's
        the learner's turn to decide, or the battle ends."""
        from ptcg.game import battle_select

        while True:
            current = obs_dict.get("current") or {}
            if current.get("result", -1) >= 0:
                return obs_dict
            if current.get("yourIndex", 0) == self._learner_index:
                return obs_dict
            obs_dict = battle_select(self._opponent_agent(obs_dict))

    def _sync_select_state(self, obs_dict):
        select = obs_dict.get("select") or {}
        options = select.get("option") or []
        if len(options) > self.MAX_OPTIONS:
            warnings.warn(
                f"Select offered {len(options)} options, exceeding "
                f"MAX_OPTIONS={self.MAX_OPTIONS}; truncating. Raise "
                "MAX_OPTIONS in obs_vectorizer.py to fix.",
                stacklevel=2,
            )
        self._n_options = min(len(options), self.MAX_OPTIONS)
        self._min_count = min(select.get("minCount", 1), self._n_options)
        self._max_count = min(select.get("maxCount", 1), self._n_options)

    def reset(self, seed=None, options=None):
        # Start a new CABT game (self-play by default; vs opponent_agent if set)
        from ptcg.game import battle_start, battle_finish

        super().reset(seed=seed)
        if self._battle is not None:
            battle_finish()
        deck = _load_deck()
        set_vectorizer_deck(deck)
        obs_dict, start_data = battle_start(deck, deck)
        if start_data.errorPlayer >= 0:
            # Invalid deck -- shouldn't happen if deck is correct
            raise RuntimeError("Battle start failed")
        if self._opponent_agent is not None:
            # Randomize which side the learner plays so it doesn't overfit
            # to always going first (or second).
            self._learner_index = int(self.np_random.integers(0, 2))
            obs_dict = self._play_opponent_until_learner_turn(obs_dict)
        self._obs = obs_dict
        self._sync_select_state(obs_dict)
        return obs_to_vector(obs_dict), {}

    def step(self, action):
        from ptcg.game import battle_select

        # `action` is a MAX_OPTIONS-length 0/1 vector. Only the first
        # n_options slots are meaningful; the rest are masked out.
        selected = [i for i in range(self._n_options) if action[i]]

        # The native engine requires exactly minCount <= len(selected) <=
        # maxCount. The mask can't enforce this jointly (each slot is an
        # independent Bernoulli), so clamp here: truncate excess picks, and
        # pad with the lowest-index unpicked options if we're short.
        if len(selected) > self._max_count:
            selected = selected[: self._max_count]
        if len(selected) < self._min_count:
            chosen = set(selected)
            for i in range(self._n_options):
                if len(selected) >= self._min_count:
                    break
                if i not in chosen:
                    selected.append(i)
                    chosen.add(i)
            selected.sort()

        obs_dict = battle_select(selected)
        if self._opponent_agent is not None:
            obs_dict = self._play_opponent_until_learner_turn(obs_dict)
        self._obs = obs_dict

        current = obs_dict.get("current") or {}
        # NOTE: result=0 means player 0 won and is falsy in Python, so this
        # must be `.get("result", -1)`, not `.get("result") or -1` (the
        # latter silently turns a player-0 win into "not done").
        result = current.get("result", -1)

        done = result >= 0
        reward = 0.0
        if done:
            if self._opponent_agent is not None:
                reward = 1.0 if result == self._learner_index else -1.0
            else:
                your_index = current.get("yourIndex", 0)
                # result=0 means player 0 wins, result=1 means player 1 wins
                reward = 1.0 if result != your_index else -1.0

        self._sync_select_state(obs_dict)
        return obs_to_vector(obs_dict), reward, done, False, {}

    def action_masks(self) -> np.ndarray:
        """Return a (MAX_OPTIONS, 2) mask for sb3-contrib's MaskablePPO.

        For each slot, column 0 is whether "not selected" is valid and
        column 1 is whether "selected" is valid. Slots beyond n_options
        don't correspond to a real option, so they're forced to 0.
        """
        mask = np.zeros((self.MAX_OPTIONS, 2), dtype=bool)
        mask[:, 0] = True
        mask[: self._n_options, 1] = True
        return mask
