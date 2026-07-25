import glob
import os
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback

from training.rewards import REWARD_TERMS


# Opponent tags that are the learner playing a copy of itself: a frozen
# league snapshot (SnapshotOpponentPool's default name, training/league.py)
# and CabtEnv's pure self-play fallback. Excluded from win_rate/overall --
# a mirror match sits near 50% by construction as learner and league improve
# together, so folding it into the aggregate both inflates "overall" and
# makes it drift for reasons unrelated to the heuristic benchmarks (a run
# whose league share differs, or whose league happens to be stronger, gets a
# different "overall" at identical heuristic skill). Each still gets its own
# win_rate/<name> series -- only the aggregate changes.
SELF_PLAY_OPPONENTS = frozenset({"league", "selfplay"})


class WinRateCallback(BaseCallback):
    """Log win rate to TensorBoard, overall and broken down per opponent.

    Reads each finished episode from Monitor's info["episode"] dict, which
    CabtEnv tags with the opponent it faced (via Monitor info_keywords). A
    win is a positive terminal reward. Counts are cumulative over the whole
    run, so each series is the agent's career win rate against that opponent
    and should trend upward as it learns; the per-opponent split shows which
    matchups it is still losing.

    "overall" aggregates the heuristic opponents ONLY -- see
    SELF_PLAY_OPPONENTS above for why the league/self-play episodes are kept
    out of it. Note this makes win_rate/overall a strictly harder number than
    it was before that exclusion, so it is not comparable with runs logged
    earlier (their overall was diluted by a ~50% mirror match).

    Scalars written (under ./ppo_cabt_logs/ via the model's tensorboard_log):
        win_rate/overall           (heuristic opponents only)
        win_rate/<opponent name>   (one per heuristic, plus "league")
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._wins: dict[str, int] = defaultdict(int)
        self._games: dict[str, int] = defaultdict(int)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue  # episode still in progress
            won = 1 if episode["r"] > 0 else 0
            opponent = episode.get("opponent", "unknown")
            self._wins[opponent] += won
            self._games[opponent] += 1
            if opponent not in SELF_PLAY_OPPONENTS:
                self._wins["overall"] += won
                self._games["overall"] += 1
        return True

    def _on_rollout_end(self) -> None:
        for key, games in self._games.items():
            if games:
                self.logger.record(f"win_rate/{key}", self._wins[key] / games)

    def summary(self) -> dict[str, tuple[int, int]]:
        """Return {opponent: (wins, games)} over the whole run.

        "overall" is included as the aggregate across the heuristic opponents
        only (see SELF_PLAY_OPPONENTS) -- so it is NOT the sum of the other
        entries, and callers wanting a true episode total must add up the
        per-opponent counts instead. Used by the end-of-training report in
        train.py.
        """
        return {key: (self._wins[key], games) for key, games in self._games.items()}


class RewardTermCallback(BaseCallback):
    """Log every reward term to TensorBoard as its mean contribution per
    episode.

    CabtEnv attaches a per-step breakdown as info["reward_terms"] (see
    training/rewards.py reward_terms); this sums them over each rollout and
    divides by the episodes that finished in it, so the numbers read in the
    same units as the terminal reward -- "this term is worth +0.27 a game"
    sits directly against the +/-1.0 win/loss and says whether it is a nudge
    or a rival objective.

    Every name in REWARD_TERMS is written every time, including the ones that
    summed to zero. That is the point: a term whose card-id or log-schema
    assumptions are wrong contributes nothing forever, and a flat 0.0 series
    says so at a glance where a missing series would not. Worth a look after
    any change to rewards.py.

    Windowed per rollout rather than cumulative over the run (unlike
    WinRateCallback), since the useful question is how a term's pull is
    changing as the policy shifts, which a career average would smear out.

    Scalars written (under ./ppo_cabt_logs/ via the model's tensorboard_log):
        reward/<term>        one per REWARD_TERMS entry, per-episode mean
        reward/shaping_total every term except "terminal", summed
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._sums: dict[str, float] = defaultdict(float)
        self._episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            terms = info.get("reward_terms")
            if terms:
                for name, value in terms.items():
                    self._sums[name] += value
            if info.get("episode") is not None:
                self._episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        # No episode finished in this rollout (long games against a small
        # n_steps): keep accumulating into the next one instead of dividing
        # by zero or logging a partial-episode figure as if it were a full one.
        if not self._episodes:
            return
        shaping_total = 0.0
        # Iterate REWARD_TERMS, not self._sums, so terms that never fired are
        # still written -- as the flat zeros that reveal them.
        for name in REWARD_TERMS:
            per_episode = self._sums.get(name, 0.0) / self._episodes
            self.logger.record(f"reward/{name}", per_episode)
            if name != "terminal":
                shaping_total += per_episode
        self.logger.record("reward/shaping_total", shaping_total)
        self._sums.clear()
        self._episodes = 0


class SnapshotCallback(BaseCallback):
    """Periodically freeze the live policy into `snapshot_dir` for league
    self-play (see training/league.py).

    Saves one snapshot at training start -- so league envs have an opponent
    from their very first episodes -- then another every `save_freq`
    timesteps, keeping only the newest `max_snapshots` files. Snapshots are
    written to a .tmp path and os.replace()d into place so workers globbing
    *.zip never see a half-written file.

    `reset` (default True) empties `snapshot_dir` up front so each run builds
    its own league from scratch. Without it a run silently inherits whatever
    the previous run left behind, which made win_rate/league incomparable
    across runs (it opened at 0.50, 0.16, 0.42 and 0.27 on four otherwise
    similar runs) and cost one run its first ~2M steps climbing out of a hole
    dug by its predecessor's final policy. Pass reset=False to deliberately
    continue a previous run's league.

    The clear happens in __init__ rather than _on_training_start because SB3
    resets the envs -- and SnapshotOpponentPool globs the directory at every
    reset -- before on_training_start fires, so a clear in that hook would
    still let the first league episodes draw stale opponents. Construct this
    callback BEFORE the VecEnv (see training/train.py).
    """

    def __init__(
        self,
        snapshot_dir: str,
        save_freq: int,
        max_snapshots: int = 5,
        reset: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.save_freq = save_freq
        self.max_snapshots = max_snapshots
        self._last_save = 0
        if reset:
            self._clear_snapshots()

    def _snapshot_files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.snapshot_dir, "*.zip")))

    def _clear_snapshots(self) -> None:
        """Delete every snapshot (plus any half-written .tmp) so this run's
        league starts empty.

        A failed delete is warned about rather than raised -- it only leaves
        one stale opponent in the pool, not a broken run -- but it does mean
        the league is no longer this run's alone, so it must not be silent.
        """
        os.makedirs(self.snapshot_dir, exist_ok=True)
        stale = self._snapshot_files() + sorted(
            glob.glob(os.path.join(self.snapshot_dir, "*.zip.tmp"))
        )
        failed = []
        for path in stale:
            try:
                os.remove(path)
            except OSError:
                failed.append(os.path.basename(path))
        if failed:
            print(
                f"[league] WARNING: could not delete {len(failed)} old "
                f"snapshot(s) in {self.snapshot_dir} -- this run's league is "
                f"contaminated by a previous run: {failed}"
            )
        elif stale and self.verbose:
            print(
                f"[league] cleared {len(stale)} snapshot(s) from "
                f"{self.snapshot_dir}"
            )

    def _on_training_start(self) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        # Empty after a reset=True clear, so this lays down the initial
        # snapshot. With reset=False a leftover league counts instead: don't
        # shadow it with a fresh-init snapshot, just start the save cadence.
        if not self._snapshot_files():
            self._save_snapshot()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._save_snapshot()
        return True

    def _save_snapshot(self) -> None:
        self._last_save = self.num_timesteps
        # Zero-padded step count so lexicographic sort == chronological.
        final = os.path.join(
            self.snapshot_dir, f"snapshot_{self.num_timesteps:012d}.zip"
        )
        tmp = final + ".tmp"
        self.model.save(tmp)
        os.replace(tmp, final)
        if self.verbose:
            print(f"[league] saved snapshot at {self.num_timesteps:,} steps")
        # Prune beyond the newest max_snapshots.
        for path in self._snapshot_files()[: -self.max_snapshots]:
            try:
                os.remove(path)
            except OSError:
                # e.g. Windows, while a worker briefly holds the file open
                # for loading -- leave it; the next prune will catch it.
                pass
