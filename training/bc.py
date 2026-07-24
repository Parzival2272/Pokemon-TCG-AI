"""Behavior-cloning (BC) warm start from the starmie heuristic.

Instead of PPO starting from a virtually random policy, pretrain the policy
net to imitate the starmie heuristic (which pilots the same DECK_PATH deck
the learner plays), then start PPO from those weights.

Two stages, both in this file:

1. Collection: the starmie expert drives CabtEnv's learner seat against the
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
    python -m training.bc                  # collect + clone with defaults
    BC_INIT=ppo_starmie_bc.zip python -m training.train

The saved zip is a normal MaskablePPO save; train.py's BC_INIT env var loads
just its policy weights into the fresh model. Since SnapshotCallback saves
an initial league snapshot at training start, the BC policy automatically
becomes the league's first frozen opponent too.
"""

import argparse
import os
import time

import numpy as np
import torch as th
import torch.nn.functional as F

from heuristics.starmie_agent import agent as starmie_agent
from training.cabt_env import CabtEnv, _sanitize_selection

# Matches train.py's PPO gamma so the value head is fit to the same return
# definition PPO will bootstrap against.
GAMMA = 0.995


# ---------------------------------------------------------------------------
# Stage 1: collect expert decisions
# ---------------------------------------------------------------------------


def collect_dataset(n_episodes, seed=0):
    """Roll `n_episodes` games with the starmie expert in the learner seat
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
                raw = starmie_agent(env._obs)
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


# ---------------------------------------------------------------------------
# Stage 2: clone
# ---------------------------------------------------------------------------


def _accuracy(policy, obs_t, mask_t, act_t, batch_size=4096):
    """Fraction of samples where the policy's argmax matches the expert."""
    hits = 0
    with th.no_grad():
        for i in range(0, len(obs_t), batch_size):
            dist = policy.get_distribution(
                obs_t[i : i + batch_size], action_masks=mask_t[i : i + batch_size]
            )
            pred = dist.distribution.probs.argmax(dim=-1)
            hits += (pred == act_t[i : i + batch_size]).sum().item()
    return hits / len(obs_t)


def behavior_clone(
    data,
    out_path,
    epochs=10,
    batch_size=256,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.0,
    device="cpu",
    seed=0,
):
    """Train a fresh MaskablePPO's policy to imitate `data`, save to
    `out_path` (a normal MaskablePPO zip, loadable by train.py's BC_INIT)."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    # Built only for its spaces + policy; never stepped or reset here.
    env = ActionMasker(CabtEnv(), lambda e: e.action_masks())
    model = MaskablePPO(
        "MlpPolicy", env, device=device, learning_rate=lr, gamma=GAMMA, seed=seed
    )
    policy = model.policy

    n = len(data["obs"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, n // 20)  # 5% holdout
    val_idx, train_idx = order[:n_val], order[n_val:]

    dev = policy.device
    obs_t = th.as_tensor(data["obs"], device=dev)
    mask_t = th.as_tensor(data["masks"], device=dev)
    act_t = th.as_tensor(data["actions"], device=dev)
    ret_t = th.as_tensor(data["returns"], device=dev)

    optimizer = th.optim.Adam(policy.parameters(), lr=lr)
    policy.set_training_mode(True)

    print(
        f"Cloning on {len(train_idx):,} samples "
        f"({n_val:,} held out), device={dev}"
    )
    baseline = _accuracy(policy, obs_t[val_idx], mask_t[val_idx], act_t[val_idx])
    print(f"  epoch  0: val accuracy {baseline:.1%} (untrained baseline)")

    for epoch in range(1, epochs + 1):
        perm = train_idx[rng.permutation(len(train_idx))]
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            values, log_prob, entropy = policy.evaluate_actions(
                obs_t[idx], act_t[idx], action_masks=mask_t[idx]
            )
            loss = (
                -log_prob.mean()
                + vf_coef * F.mse_loss(values.flatten(), ret_t[idx])
                - ent_coef * entropy.mean()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc = _accuracy(policy, obs_t[val_idx], mask_t[val_idx], act_t[val_idx])
        print(
            f"  epoch {epoch:2d}: loss {total_loss / n_batches:.4f}, "
            f"val accuracy {val_acc:.1%}"
        )

    policy.set_training_mode(False)
    model.save(out_path)
    env.close()
    print(f"Saved BC-initialized model to {out_path}.zip")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Behavior-clone the starmie heuristic into a MaskablePPO "
        "policy (see module docstring)."
    )
    # Collection runs ~30 games/s single-process, so 2000 episodes (~80k
    # samples) is well under two minutes.
    parser.add_argument("--episodes", type=int, default=2000, help="games to collect")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--dataset",
        default="bc_dataset.npz",
        help="dataset cache; loaded if it exists (delete to recollect)",
    )
    parser.add_argument("--out", default="ppo_starmie_bc", help="model save path")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.dataset):
        print(f"Loading cached dataset {args.dataset}")
        with np.load(args.dataset) as npz:
            data = {k: npz[k] for k in npz.files}
    else:
        print(f"Collecting {args.episodes} expert episodes...")
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
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
