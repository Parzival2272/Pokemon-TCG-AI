"""Behavior-cloning (BC) warm start from the crustle heuristic.

Instead of PPO starting from a virtually random policy, pretrain the policy
net to imitate the crustle heuristic (which pilots the same DECK_PATH deck
the learner plays), then start PPO from those weights.

Two stages, both in this file:

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
    python -m training.bc                  # collect + clone with defaults
    BC_INIT=ppo_crustle_bc.zip python -m training.train

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

from heuristics.crustle_agent import agent as crustle_agent
from training.cabt_env import POLICY_NET_ARCH, CabtEnv, _sanitize_selection

# Matches train.py's PPO gamma so the value head is fit to the same return
# definition PPO will bootstrap against.
GAMMA = 0.995


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
    weight_decay=1e-4,
    patience=8,
    device="cpu",
    seed=0,
):
    """Train a fresh MaskablePPO's policy to imitate `data`, save to
    `out_path` (a normal MaskablePPO zip, loadable by train.py's BC_INIT).

    Saves the BEST-val-accuracy weights, not the final epoch's: the training
    policy loss keeps falling long after val accuracy plateaus (overfitting),
    so the last epoch is the most overfit. Stops early if val accuracy hasn't
    improved for `patience` epochs. AdamW's `weight_decay` adds mild L2
    regularization to widen the usable epoch window before overfit sets in.
    """
    import copy

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    # Built only for its spaces + policy; never stepped or reset here.
    env = ActionMasker(CabtEnv(), lambda e: e.action_masks())
    # net_arch MUST match train.py's PPO model: BC_INIT warm-starts by a strict
    # load_state_dict, so both read the same POLICY_NET_ARCH constant.
    model = MaskablePPO(
        "MlpPolicy",
        env,
        device=device,
        learning_rate=lr,
        gamma=GAMMA,
        seed=seed,
        policy_kwargs=dict(net_arch=list(POLICY_NET_ARCH)),
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

    optimizer = th.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    policy.set_training_mode(True)

    print(
        f"Cloning on {len(train_idx):,} samples "
        f"({n_val:,} held out), device={dev}"
    )
    baseline = _accuracy(policy, obs_t[val_idx], mask_t[val_idx], act_t[val_idx])
    print(f"  epoch  0: val accuracy {baseline:.1%} (untrained baseline)")

    # Keep the best-val weights (CPU copy) so we can restore them before saving.
    best_val = baseline
    best_state = copy.deepcopy(policy.state_dict())
    best_epoch = 0
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        perm = train_idx[rng.permutation(len(train_idx))]
        # Track the imitation (policy NLL) and value-fit (MSE) terms
        # separately: the total loss can't go near 0 because the value head is
        # regressing noisy discounted returns, so only the policy term reflects
        # imitation quality -- and even that has a floor set by observation
        # aliasing and the expert's arbitrary score tie-breaking.
        total_loss = total_pg = total_vf = 0.0
        n_batches = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            values, log_prob, entropy = policy.evaluate_actions(
                obs_t[idx], act_t[idx], action_masks=mask_t[idx]
            )
            pg_loss = -log_prob.mean()
            vf_loss = F.mse_loss(values.flatten(), ret_t[idx])
            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            n_batches += 1
        val_acc = _accuracy(policy, obs_t[val_idx], mask_t[val_idx], act_t[val_idx])
        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            best_state = copy.deepcopy(policy.state_dict())
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        print(
            f"  epoch {epoch:2d}: loss {total_loss / n_batches:.4f} "
            f"(policy {total_pg / n_batches:.4f}, value {total_vf / n_batches:.4f}), "
            f"val accuracy {val_acc:.1%}{'  <- best' if improved else ''}"
        )
        if epochs_since_best >= patience:
            print(
                f"  early stop: no val improvement in {patience} epochs "
                f"(best {best_val:.1%} @ epoch {best_epoch})"
            )
            break

    # Restore and save the best-val weights, not the (more overfit) last epoch.
    policy.load_state_dict(best_state)
    policy.set_training_mode(False)
    model.save(out_path)
    env.close()
    print(
        f"Saved BC-initialized model (best val {best_val:.1%} @ epoch "
        f"{best_epoch}) to {out_path}.zip"
    )
    return model


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
    parser.add_argument(
        "--dataset",
        default="bc_dataset.npz",
        help="dataset cache; loaded if it exists (delete to recollect)",
    )
    parser.add_argument("--out", default="ppo_crustle_bc", help="model save path")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cpu"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
