import glob
import os
from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback


class WinRateCallback(BaseCallback):
    """Log win rate to TensorBoard, overall and broken down per opponent.

    Reads each finished episode from Monitor's info["episode"] dict, which
    CabtEnv tags with the opponent it faced (via Monitor info_keywords). A
    win is a positive terminal reward. Counts are cumulative over the whole
    run, so each series is the agent's career win rate against that opponent
    and should trend upward as it learns; the per-opponent split shows which
    matchups it is still losing.

    Scalars written (under ./ppo_cabt_logs/ via the model's tensorboard_log):
        win_rate/overall
        win_rate/<opponent name>   (one per heuristic, plus "selfplay")
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
            self._wins["overall"] += won
            self._games["overall"] += 1
        return True

    def _on_rollout_end(self) -> None:
        for key, games in self._games.items():
            if games:
                self.logger.record(f"win_rate/{key}", self._wins[key] / games)

    def summary(self) -> dict[str, tuple[int, int]]:
        """Return {opponent: (wins, games)} over the whole run.

        "overall" is included as the aggregate across every opponent. Used by
        the end-of-training report in train.py.
        """
        return {key: (self._wins[key], games) for key, games in self._games.items()}


class SnapshotCallback(BaseCallback):
    """Periodically freeze the live policy into `snapshot_dir` for league
    self-play (see training/league.py).

    Saves one snapshot at training start -- so league envs have an opponent
    from their very first episodes -- then another every `save_freq`
    timesteps, keeping only the newest `max_snapshots` files. Snapshots are
    written to a .tmp path and os.replace()d into place so workers globbing
    *.zip never see a half-written file.
    """

    def __init__(
        self,
        snapshot_dir: str,
        save_freq: int,
        max_snapshots: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.snapshot_dir = snapshot_dir
        self.save_freq = save_freq
        self.max_snapshots = max_snapshots
        self._last_save = 0

    def _snapshot_files(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.snapshot_dir, "*.zip")))

    def _on_training_start(self) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)
        # A leftover league from a previous run counts: don't shadow it with
        # a fresh-init snapshot, just start the save cadence.
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
