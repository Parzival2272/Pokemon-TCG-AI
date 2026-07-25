"""Frozen-checkpoint ("league") opponents for self-play training.

SnapshotCallback (training/callbacks.py) periodically freezes the live policy
into a snapshot directory; SnapshotOpponentPool turns that directory into a
CabtEnv opponent pool that is re-scanned at every episode reset, so snapshots
saved mid-run join the league without restarting workers.

Why frozen checkpoints instead of both-sides-live self-play: with both sides
of one game feeding a single sequential rollout stream, consecutive
observations alternate perspective, so PPO's GAE bootstraps V(s_{t+1}) from
the *opponent's* view of the board -- an (approximately) inverted estimate --
and the losing side never sees its -1 terminal (the game ends on the winner's
transition). Against a frozen snapshot the learner owns every transition, so
the stream is a clean single-perspective MDP, while the pool of past selves
still provides self-play-style curriculum and guards against strategy
cycling.

Snapshot opponents pilot the learner's own deck (a mirror match): that's the
deck they were trained on, and it keeps obs_vectorizer's global deck (set to
the learner's at reset) correct for both seats.
"""

import glob
import os

import numpy as np

from training.obs_vectorizer import MAX_OPTIONS, obs_to_vector

# Mirrors CabtEnv.STOP_ACTION (kept literal here to avoid importing the env
# into every worker just for one constant).
STOP_ACTION = MAX_OPTIONS


def _load_policy(path):
    """Load just the policy net from an SB3 zip.

    The surrounding algorithm object is dropped immediately: MaskablePPO.load
    also reconstructs the rollout buffer (n_steps x n_envs x OBS_SIZE floats,
    easily 100s of MB), which every worker would otherwise keep per cached
    snapshot. The bare MaskableActorCriticPolicy is a tiny MLP.
    """
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(path, device="cpu")
    policy = model.policy
    del model
    return policy


class _SnapshotAgent:
    """CabtEnv-style opponent -- callable(obs_dict) -> list[int] -- piloting a
    frozen policy.

    Replays CabtEnv's autoregressive action decoding: pick one option at a
    time (re-vectorizing the obs with the partial `picked` selection, same
    mask rules as CabtEnv.action_masks) until the policy chooses STOP or
    maxCount is reached.
    """

    def __init__(self, policy):
        self._policy = policy

    def __call__(self, obs_dict):
        select = obs_dict.get("select") or {}
        n = min(len(select.get("option") or []), MAX_OPTIONS)
        max_count = min(select.get("maxCount", 1), n)
        min_count = min(select.get("minCount", 1), max_count)
        picked: list[int] = []
        while len(picked) < max_count:
            mask = np.zeros(MAX_OPTIONS + 1, dtype=bool)
            mask[:n] = True
            mask[picked] = False
            if len(picked) >= min_count:
                mask[STOP_ACTION] = True
            # Sample (deterministic=False) so the league offers varied play
            # rather than one exploitable line per snapshot.
            action, _ = self._policy.predict(
                obs_to_vector(obs_dict, picked=picked),
                action_masks=mask,
                deterministic=False,
            )
            action = int(action)
            if action == STOP_ACTION:
                break
            picked.append(action)
        return picked


class SnapshotOpponentPool:
    """Zero-arg callable opponent pool over a directory of policy snapshots.

    CabtEnv calls it at each reset(): it globs `snapshot_dir` for *.zip, so
    snapshots saved (or pruned) mid-run are picked up automatically. Loaded
    policies are cached per worker process; cache entries whose file was
    pruned are dropped. All entries share one `name` so WinRateCallback logs
    a single win_rate/<name> series (expected to hover near 50% as learner
    and league improve together).

    `fallback` is an ordinary (name, agent_fn, deck) pool used for episodes
    where no snapshot could be loaded. That window is small but real: SB3
    resets the envs inside _setup_learn() BEFORE firing on_training_start(),
    which is where SnapshotCallback writes the first snapshot, so each league
    worker's very first episode of a run finds an empty directory. Without a
    fallback CabtEnv treats the empty pool as pure self-play -- exactly the
    both-sides-live mode the module docstring above explains this class exists
    to avoid, since those transitions get mis-credited by GAE. It is only
    ~N_LEAGUE_ENVS episodes per run, but they are wrong rather than merely
    few, and they also spawn a stray win_rate/selfplay series. Passing the
    heuristic pool here makes them ordinary heuristic games instead. Leave it
    None to keep the old self-play fallback.

    Picklable before first use (no torch state until a snapshot is actually
    loaded), so it survives the trip to SubprocVecEnv workers.
    """

    def __init__(self, snapshot_dir, deck, name="league", fallback=None):
        self.snapshot_dir = snapshot_dir
        self.deck = deck
        self.name = name
        self.fallback = list(fallback) if fallback else []
        self._agents: dict[str, _SnapshotAgent] = {}

    def __call__(self):
        files = sorted(glob.glob(os.path.join(self.snapshot_dir, "*.zip")))
        for stale in set(self._agents) - set(files):
            del self._agents[stale]
        entries = []
        for path in files:
            agent = self._agents.get(path)
            if agent is None:
                try:
                    agent = _SnapshotAgent(_load_policy(path))
                except Exception:
                    # File pruned (or otherwise unreadable) between the glob
                    # and the load; skip it until the next reset.
                    continue
                self._agents[path] = agent
            entries.append((self.name, agent, self.deck))
        # Keyed on `entries`, not `files`, so a directory whose snapshots all
        # failed to load falls back too rather than going self-play.
        return entries or self.fallback
