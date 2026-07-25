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

# Shared policy/value network architecture. bc.py builds a MaskablePPO with
# this net_arch, and train.py warm-starts (load_state_dict, which is strict)
# into a model that must have the SAME arch -- so both import this one constant
# to stay in lock-step. The enriched observation (see obs_vectorizer.py) is
# wider and carries the card-semantic features the heuristic branches on, so a
# 64x64 net is underpowered; 256x256 gives it room without materially slowing
# CPU training (vectorization, not the forward pass, is the hot path here).
POLICY_NET_ARCH = [256, 256]


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
    # Action index MAX_OPTIONS is the "stop / finish selection" action; indices
    # 0..MAX_OPTIONS-1 pick the corresponding option.
    STOP_ACTION = MAX_OPTIONS

    def __init__(self, opponent_agents=None):
        """
        Args:
            opponent_agents: Optional pool of opponents. Either None
                (pure self-play -- both sides are the RL policy, both playing
                the DECK_PATH deck), a list of (name, agent_fn, deck) tuples,
                or a zero-arg callable returning such a list. A callable is
                re-evaluated at every reset(), so the pool can change mid-run
                (e.g. league snapshots saved while training -- see
                training/league.py); if it returns an empty list the episode
                falls back to pure self-play.
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
        # Autoregressive selection: each step picks ONE option (0..MAX_OPTIONS-1)
        # or STOP_ACTION to finish. A decision needing k picks (discard 2, choose
        # 3 prizes, ...) is spread over k steps, and STOP finalizes once the
        # count is legal. This replaces the old MultiBinary(MAX_OPTIONS), where
        # the policy couldn't enforce the required count jointly and the env had
        # to clamp the sampled subset after the fact -- training PPO on a
        # different action than was executed. A single n-way choice is now one
        # masked categorical instead of MAX_OPTIONS independent Bernoullis.
        self.action_space = gym.spaces.Discrete(self.MAX_OPTIONS + 1)

        self._opponent_agents = opponent_agents
        # Per-episode opponent, chosen in reset() (None while in self-play).
        self._opponent_agent = None
        self._opponent_name = "selfplay"
        self._learner_index = 0
        self._obs = None
        self._n_options = 0
        self._min_count = 0
        self._max_count = 0
        # Options picked so far in the current decision (reset at each decision).
        self._picked: list[int] = []

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
        self._max_count = min(select.get("maxCount", 1), self._n_options)
        # Clamp min <= max so the STOP action is always reachable (guards
        # against any odd engine data where minCount > available options).
        self._min_count = min(select.get("minCount", 1), self._max_count)
        # Starting a fresh decision: nothing picked yet.
        self._picked = []

    def reset(self, seed=None, options=None):
        # Start a new CABT game (self-play by default; vs opponent_agent if set)
        from ptcg.game import battle_start, battle_finish
        from ptcg.sim import Battle

        super().reset(seed=seed)
        # Free the previous episode's native battle before starting a new one,
        # or it leaks inside cg.dll. Battle.battle_ptr is the engine's own
        # record of the live battle (one per process, see train.py), so it --
        # not env-local state -- is the thing to check.
        if Battle.battle_ptr:
            battle_finish()
            Battle.battle_ptr = None
        learner_deck = _load_deck()
        # Prize-belief features are always from the learner's perspective, so
        # the vectorizer deck is the learner's deck regardless of opponent.
        set_vectorizer_deck(learner_deck)

        # A callable pool is re-evaluated every episode so it can grow/shrink
        # mid-run (league snapshots); a plain list is used as-is.
        pool = (
            self._opponent_agents()
            if callable(self._opponent_agents)
            else self._opponent_agents
        )
        if pool:
            # Pick a fresh opponent for this episode; it pilots its own deck.
            self._opponent_name, self._opponent_agent, opp_deck = pool[
                int(self.np_random.integers(0, len(pool)))
            ]
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
        return obs_to_vector(obs_dict, picked=self._picked), {}

    def step(self, action):
        from ptcg.game import battle_select

        action = int(action)

        # Decide whether this action finalizes the current selection or just
        # adds one more pick to it.
        if action == self.STOP_ACTION:
            finalize = True
        else:
            # A pick. The mask should already forbid illegal picks, but guard
            # anyway (out-of-range, duplicate, or past maxCount).
            if (
                0 <= action < self._n_options
                and action not in self._picked
                and len(self._picked) < self._max_count
            ):
                self._picked.append(action)
            # maxCount reached -> the selection is complete, finalize now
            # (no explicit STOP needed; keeps single-pick decisions one step).
            finalize = len(self._picked) >= self._max_count

        if not finalize:
            # Intermediate sub-step: the engine has NOT advanced, so the board
            # is unchanged and only the partial selection differs. No reward,
            # not done -- the decision continues on the next step.
            return (
                obs_to_vector(self._obs, picked=self._picked),
                0.0,
                False,
                False,
                {},
            )

        # Learner's observation before the selection is submitted -- the
        # "before" state for prize-differential shaping. The board is identical
        # across this decision's sub-steps, so self._obs (the decision's start
        # state) is the correct baseline.
        prev_obs = self._obs

        selected = _sanitize_selection(
            self._picked, self._n_options, self._min_count, self._max_count
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
        # The reward for this decision belongs to the player who just acted. The
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
        return obs_to_vector(obs_dict, picked=self._picked), reward, done, False, info

    def close(self):
        from ptcg.game import battle_finish
        from ptcg.sim import Battle

        if Battle.battle_ptr:
            battle_finish()
            Battle.battle_ptr = None
        super().close()

    def action_masks(self) -> np.ndarray:
        """Return a (MAX_OPTIONS + 1,) boolean mask for MaskablePPO's Discrete
        action space.

        A real option is legal to pick iff it isn't already picked and the
        selection isn't full yet (len(picked) < maxCount). STOP is legal once at
        least minCount options are picked. At least one action is always legal
        when the env queries the policy: it only queries while
        len(picked) < maxCount <= n_options, so some unpicked option remains.
        """
        mask = np.zeros(self.MAX_OPTIONS + 1, dtype=bool)
        if len(self._picked) < self._max_count:
            for i in range(self._n_options):
                if i not in self._picked:
                    mask[i] = True
        if len(self._picked) >= self._min_count:
            mask[self.STOP_ACTION] = True
        return mask
