import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from training.obs_vectorizer import (
    MAX_OPTIONS,
    VECTOR_SIZE,
    obs_to_vector,
    set_vectorizer_deck,
)
from training.rewards import compute_reward
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


def _sanitize_selection(picks, n_options, min_count, max_count):
    """Coerce a raw selection into one the native engine will accept.

    Keeps only unique, in-range indices, then clamps the count into
    [min_count, max_count] -- truncating extras and padding with the
    lowest-index unpicked options when short. Used for both the learner
    (whose action is already in range) and heuristic opponents, which can
    occasionally return a malformed selection (e.g. too few discards) that
    kaggle_environments would sanitize but raw battle_select rejects.
    """
    seen = set()
    valid = []
    for i in picks or []:
        if isinstance(i, int) and 0 <= i < n_options and i not in seen:
            seen.add(i)
            valid.append(i)
    if len(valid) > max_count:
        valid = valid[:max_count]
    if len(valid) < min_count:
        for i in range(n_options):
            if len(valid) >= min_count:
                break
            if i not in seen:
                seen.add(i)
                valid.append(i)
        valid.sort()
    return valid


class CabtEnv(gym.Env):
    """Wraps a CABT game as a Gymnasium environment."""

    OBS_SIZE = VECTOR_SIZE
    MAX_OPTIONS = MAX_OPTIONS

    def __init__(self, opponent_agents=None):
        """
        Args:
            opponent_agents: Optional pool of heuristic opponents. Either None
                (pure self-play -- both sides are the RL policy, both playing
                the DECK_PATH deck) or a list of (name, agent_fn, deck) tuples.
                When a pool is given, one opponent is chosen uniformly at
                random each episode; it controls one side (the learner's side
                is randomized) and pilots its own `deck`, while the learner
                always plays the DECK_PATH deck. `agent_fn` is a
                callable(obs_dict) -> list[int]; `name` labels the matchup for
                logging; `deck` is a 60-card ID list.
        """
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(self.OBS_SIZE,), dtype=np.float32
        )
        # A subset of the MAX_OPTIONS slots is selected per step, since some
        # decisions require choosing 0, 2, or any other count of options at
        # once (see SelectData.minCount/maxCount), not just exactly 1.
        self.action_space = gym.spaces.MultiBinary(self.MAX_OPTIONS)

        self._opponent_agents = opponent_agents
        # Per-episode opponent, chosen in reset() (None while in self-play).
        self._opponent_agent = None
        self._opponent_name = "selfplay"
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
            select = obs_dict.get("select") or {}
            n = len(select.get("option") or [])
            min_count = min(select.get("minCount", 1), n)
            max_count = min(select.get("maxCount", 1), n)
            # Some heuristics can raise on certain board states (e.g. an
            # out-of-range index in their discard logic). kaggle_environments
            # swallows such agent errors; do the same here so one opponent's
            # bug can't kill a long training run -- fall back to a safe
            # lowest-index selection (sanitize pads it to minCount).
            try:
                raw = self._opponent_agent(obs_dict)
            except Exception:
                raw = []
            obs_dict = battle_select(
                _sanitize_selection(raw, n, min_count, max_count)
            )

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
        learner_deck = _load_deck()
        # Prize-belief features are always from the learner's perspective, so
        # the vectorizer deck is the learner's deck regardless of opponent.
        set_vectorizer_deck(learner_deck)

        if self._opponent_agents:
            # Pick a fresh opponent for this episode; it pilots its own deck.
            self._opponent_name, self._opponent_agent, opp_deck = (
                self._opponent_agents[
                    int(self.np_random.integers(0, len(self._opponent_agents)))
                ]
            )
            # Randomize which side the learner plays so it doesn't overfit
            # to always going first (or second).
            self._learner_index = int(self.np_random.integers(0, 2))
            if self._learner_index == 0:
                obs_dict, start_data = battle_start(learner_deck, opp_deck)
            else:
                obs_dict, start_data = battle_start(opp_deck, learner_deck)
        else:
            # Pure self-play: both sides are the RL policy on the same deck.
            self._opponent_name, self._opponent_agent = "selfplay", None
            obs_dict, start_data = battle_start(learner_deck, learner_deck)

        if start_data.errorPlayer >= 0:
            # Invalid deck -- shouldn't happen if decks are correct
            raise RuntimeError("Battle start failed")
        if self._opponent_agent is not None:
            obs_dict = self._play_opponent_until_learner_turn(obs_dict)
        self._obs = obs_dict
        self._sync_select_state(obs_dict)
        return obs_to_vector(obs_dict), {}

    def step(self, action):
        from ptcg.game import battle_select

        # Learner's observation before this action -- the "before" state for
        # prize-differential shaping (in heuristic mode it's always the
        # learner's perspective, captured before self._obs is overwritten).
        prev_obs = self._obs

        # `action` is a MAX_OPTIONS-length 0/1 vector. Only the first
        # n_options slots are meaningful; the rest are masked out. The native
        # engine requires exactly minCount <= len(selected) <= maxCount, which
        # the per-slot Bernoulli mask can't enforce jointly, so clamp here.
        selected = [i for i in range(self._n_options) if action[i]]
        selected = _sanitize_selection(
            selected, self._n_options, self._min_count, self._max_count
        )

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
        # The reward for this step belongs to the player who just acted. The
        # rollout pairs it with prev_obs (the state the action was chosen
        # from), so "me" is that player, and compute_reward reads prizes /
        # decides win-loss from their (absolute) index.
        if self._opponent_agent is not None:
            me_index = self._learner_index  # learner is a fixed side
        else:
            # Self-play: the actor is whoever was to move in prev_obs. (Using
            # the *terminal* obs's yourIndex here is wrong -- control doesn't
            # flip at game end, so result == that index and the winning move
            # would score -1.)
            me_index = (prev_obs.get("current") or {}).get("yourIndex", 0)
        reward = compute_reward(prev_obs, obs_dict, done, result, me_index)

        # Tag the finished game with the opponent so WinRateCallback can
        # bucket win rate per matchup (Monitor lifts this into info["episode"]
        # via info_keywords=("opponent",)).
        info = {"opponent": self._opponent_name} if done else {}

        self._sync_select_state(obs_dict)
        return obs_to_vector(obs_dict), reward, done, False, info

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
