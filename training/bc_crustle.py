"""Behavior-cloning (BC) warm start from the crustle heuristic.

The crustle counterpart to training/bc.py (which clones the starmie expert).
Same two stages and the same cloning code -- `behavior_clone` is imported from
bc.py rather than copied, so the two passes can never drift on net_arch, the
best-val checkpointing, or the value-head fit. Only the expert, the deck it is
checked against, and the artifact names differ.

Instead of PPO starting from a virtually random policy, pretrain the policy
net to imitate the crustle heuristic (which pilots the same DECK_PATH deck the
learner plays), then start PPO from those weights.

Why a deck check runs first: the expert must be the heuristic written for
whatever cabt_env.DECK_PATH currently points at. A heuristic piloting a foreign
list demonstrates lines its deck cannot support -- the policy would be warm
started into confident nonsense, which is worse than starting random. bc.py has
the same coupling to the starmie list, so exactly one of the two files is
runnable at a time, decided by DECK_PATH; _assert_crustle_deck below makes that
a loud failure instead of a silently bad dataset.

Two stages, both inherited from bc.py's design:

1. Collection: the crustle expert drives CabtEnv's learner seat against the
   usual heuristic OPPONENT_POOL. Each engine decision (a list of option
   indices) is decoded into the env's autoregressive action space -- one
   sample per pick, plus a STOP sample when the expert stops short of
   maxCount -- by literally feeding the picks through env.step(), so the
   recorded (obs, mask) pairs are exactly what the policy will see in
   training, including partial-selection sub-steps.

2. Cloning: masked cross-entropy on the expert's actions through
   MaskableActorCriticPolicy.evaluate_actions, plus an MSE term fitting the
   value head to the observed discounted returns (gamma matches train.py)
   so PPO's first advantage estimates aren't pure noise.

Usage:
    python -m training.bc_crustle           # collect + clone with defaults
    BC_INIT=ppo_crustle_bc.zip python -m training.train

The saved zip is a normal MaskablePPO save; train.py's BC_INIT env var loads
just its policy weights into the fresh model. Since SnapshotCallback saves an
initial league snapshot at training start, the BC policy automatically becomes
the league's first frozen opponent too.
"""

import argparse
import os
import time

import numpy as np

from heuristics.crustle_agent import agent as crustle_agent
from heuristics.crustle_agent import set_deck as crustle_set_deck
from training.bc import GAMMA, behavior_clone
from training.cabt_env import DECK_PATH, CabtEnv, _load_deck, _sanitize_selection
from training.crustle_rewards import CRUSTLE_LINE_IDS


def _assert_crustle_deck(deck):
    """Fail loudly if the learner seat isn't actually piloting the crustle list.

    Cheap structural check rather than an exact-list comparison, so tuning a
    few slots in crustle_deck.csv doesn't break the BC pass -- but pointing
    DECK_PATH back at the starmie list (which has no Dwebble/Crustle at all)
    does, which is the mistake worth catching.
    """
    if not any(cid in CRUSTLE_LINE_IDS for cid in deck):
        raise SystemExit(
            f"cabt_env.DECK_PATH ({DECK_PATH}) has no Dwebble/Crustle in it, so "
            f"the crustle heuristic cannot pilot it. Point DECK_PATH at the "
            f"crustle list, or run training/bc.py if you meant the starmie deck."
        )


# ---------------------------------------------------------------------------
# Stage 1: collect expert decisions
# ---------------------------------------------------------------------------


def collect_dataset(n_episodes, seed=0):
    """Roll `n_episodes` games with the crustle expert in the learner seat
    (vs the train.py heuristic pool) and return BC arrays.

    Returns dict with:
        obs      float32 (N, OBS_SIZE)   observation vectors
        masks    bool    (N, MAX_OPTIONS + 1) legal-action masks
        actions  int64   (N,)            expert action (pick index or STOP)
        returns  float32 (N,)            discounted return-to-go, GAMMA
    """
    # Imported here, not at module top: train.py computes its env/device
    # config at import time, which collection needs (the pool) but a
    # clone-only run (cached dataset) does not.
    from training.train import OPPONENT_POOL

    learner_deck = _load_deck()
    _assert_crustle_deck(learner_deck)
    # crustle_agent caches a module-level `deck` at import time, read from the
    # deck.csv shipped in its own package. That is already the crustle list, so
    # this is now belt-and-braces rather than a fix -- but DECK_PATH is what
    # the learner actually plays, and pinning the expert to the same list keeps
    # the two from drifting if either file is edited.
    crustle_set_deck(learner_deck)

    env = CabtEnv(opponent_agents=OPPONENT_POOL)
    obs_list, mask_list, act_list = [], [], []
    returns_list = []
    wins = 0

    for episode in range(n_episodes):
        obs_vec, _ = env.reset(seed=seed + episode)
        ep_rewards = []
        done = False
        while not done:
            # A fresh decision: env always finalizes before control returns
            # here, so _picked is empty and _sync_select_state has run.
            try:
                raw = crustle_agent(env._obs)
            except Exception:
                raw = []
            selected = _sanitize_selection(
                raw, env._n_options, env._min_count, env._max_count
            )
            # Decode the selection into autoregressive actions exactly as the
            # policy will produce them: each pick in the expert's order, then
            # STOP unless maxCount auto-finalizes.
            actions = list(selected)
            if len(actions) < env._max_count:
                actions.append(CabtEnv.STOP_ACTION)
            for action in actions:
                mask = env.action_masks()
                assert mask[action], (
                    f"expert action {action} illegal under env mask "
                    f"(n={env._n_options}, min={env._min_count}, "
                    f"max={env._max_count}, picked={env._picked})"
                )
                obs_list.append(obs_vec)
                mask_list.append(mask)
                act_list.append(action)
                obs_vec, reward, done, _, info = env.step(action)
                ep_rewards.append(reward)
                if done:
                    break
        if ep_rewards and ep_rewards[-1] > 0:
            wins += 1
        # Discounted return-to-go for the value head.
        ret = 0.0
        ep_returns = [0.0] * len(ep_rewards)
        for i in range(len(ep_rewards) - 1, -1, -1):
            ret = ep_rewards[i] + GAMMA * ret
            ep_returns[i] = ret
        returns_list.extend(ep_returns)
        assert len(returns_list) == len(obs_list), "sample/return misalignment"
        if (episode + 1) % 25 == 0:
            print(
                f"  collected {episode + 1}/{n_episodes} episodes "
                f"({len(obs_list):,} samples, expert win rate "
                f"{wins / (episode + 1):.0%})"
            )

    env.close()
    return {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "masks": np.asarray(mask_list, dtype=bool),
        "actions": np.asarray(act_list, dtype=np.int64),
        "returns": np.asarray(returns_list, dtype=np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Behavior-clone the crustle heuristic into a MaskablePPO "
        "policy (see module docstring)."
    )
    # Collection runs ~30 games/s single-process. More data is the main lever
    # against the val plateau (train loss kept falling while val went flat), so
    # default higher than the original 2000; ~6000 games is still a few minutes.
    parser.add_argument("--episodes", type=int, default=6000, help="games to collect")
    # Epochs is now an upper bound: training early-stops on val plateau and
    # saves the best-val checkpoint, so a high cap just gives it room.
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience", type=int, default=8, help="early-stop patience (epochs)"
    )
    # Deliberately NOT bc.py's "bc_dataset.npz": that cache holds starmie
    # expert decisions of identical obs width, so sharing the filename would
    # sail past the width check below and clone the wrong expert.
    parser.add_argument(
        "--dataset",
        default="bc_dataset_crustle.npz",
        help="dataset cache; loaded if it exists (delete to recollect)",
    )
    parser.add_argument("--out", default="ppo_crustle_bc", help="model save path")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Checked before the (multi-minute) collection, and again inside
    # collect_dataset for callers that skip main().
    _assert_crustle_deck(_load_deck())

    data = None
    if os.path.exists(args.dataset):
        print(f"Loading cached dataset {args.dataset}")
        with np.load(args.dataset) as npz:
            data = {k: npz[k] for k in npz.files}
        # The observation layout changed (obs_vectorizer.py), so a cache from an
        # older width would feed wrong-sized vectors into the new policy net.
        # Detect the mismatch and re-collect rather than crash cryptically.
        if data["obs"].shape[1] != CabtEnv.OBS_SIZE:
            print(
                f"  cached obs width {data['obs'].shape[1]} != current OBS_SIZE "
                f"{CabtEnv.OBS_SIZE} (vectorizer changed); re-collecting."
            )
            data = None

    if data is None:
        print(f"Collecting {args.episodes} crustle expert episodes...")
        start = time.perf_counter()
        data = collect_dataset(args.episodes, seed=args.seed)
        print(
            f"Collected {len(data['obs']):,} samples in "
            f"{time.perf_counter() - start:.0f}s; caching to {args.dataset}"
        )
        np.savez_compressed(args.dataset, **data)

    behavior_clone(
        data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
