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
