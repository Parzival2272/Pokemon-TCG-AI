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
from training.cabt_env import (
    DECK_PATH,
    POLICY_NET_ARCH,
    CabtEnv,
    _load_deck,
    _sanitize_selection,
)

# Matches train.py's PPO gamma so the value head is fit to the same return
# definition PPO will bootstrap against.
GAMMA = 0.995


# ---------------------------------------------------------------------------
# Stage 1: collect expert decisions
# ---------------------------------------------------------------------------


def collect_dataset(n_episodes, seed=0):
    """Roll `n_episodes` games with the crustle expert in the learner seat and
    return BC arrays -- collect_expert_dataset with this file's expert."""
    return collect_expert_dataset(crustle_agent, n_episodes, seed=seed)


def collect_expert_dataset(
    expert, n_episodes, seed=0, deck_path=None, reward_module=None
):
    """Roll `n_episodes` games with `expert` in the learner seat (vs the
    train.py heuristic pool) and return BC arrays.

    Args:
        expert: callable(obs_dict) -> list[int], a heuristic agent.
        deck_path: deck the learner seat pilots, default cabt_env.DECK_PATH.
            MUST be the list `expert` was written for -- an expert driving a
            foreign deck demonstrates lines its deck cannot support, and the
            policy gets warm started into confident nonsense, which is worse
            than starting random.
        reward_module: reward module for the value targets, default
            cabt_env's (crustle_rewards). Swap it together with deck_path --
            see CabtEnv's constructor docstring and training/outcome_rewards.

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

    env = CabtEnv(
        opponent_agents=OPPONENT_POOL,
        deck_path=deck_path,
        reward_module=reward_module,
    )
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
                raw = expert(env._obs)
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

    On `device`: prefer "cuda" here if you have a GPU, and do NOT carry over
    train.py's conclusion that cuda is 8x SLOWER. That measurement is about
    PPO *rollout collection*, which calls the policy once per env step --
    thousands of tiny sequential forward passes, where per-call kernel-launch
    and sync overhead dwarfs the compute. Cloning is the opposite shape:
    batched supervised passes over a tensor that is uploaded to the device
    once (below) and never leaves it, which is what a GPU is for.

    Measured on this repo, cloning the 416k-sample iono dataset (RTX 5070 Ti
    vs a 16-core CPU): ~4x faster per epoch, and bit-identical -- the same
    seed gives the same permutation, so both devices reproduced the same
    per-epoch losses and val accuracies to 4 decimals. The whole dataset sits
    in VRAM, so the ceiling is memory, not speed: at OBS_SIZE=3312 the obs
    tensor alone is ~13 KB/sample (416k samples ~= 5.5 GiB), so a dataset
    much past ~1M samples wants a bigger card or a batched upload.
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


# ---------------------------------------------------------------------------
# Shared CLI for the per-expert clone passes
# ---------------------------------------------------------------------------


def _auto_device(data):
    """Pick the device to clone `data` on: "cuda" when it will fit, else "cpu".

    Unlike PPO rollout collection -- where train.py measured cuda 8x SLOWER --
    cloning is a batched supervised loop and a GPU wins outright (~4x, with
    bit-identical results; see behavior_clone's docstring). So it is the
    default rather than something you have to know to ask for.

    The catch is that behavior_clone uploads the WHOLE dataset to the device
    and keeps it there, so on a small card "cuda" turns a working CPU run into
    an out-of-memory crash -- paid for only after collection has finished. Ask
    the driver for actual free VRAM and fall back rather than risk that. The
    1.3x covers the eval-batch and autograd allocations on top of the dataset;
    the extra GiB covers the model, its AdamW state, and the CUDA context.

    An explicit DEVICE env var (or --device) overrides all of this, including
    to force "cuda" on a card this would rule out.
    """
    override = os.environ.get("DEVICE")
    if override:
        return override
    if not th.cuda.is_available():
        return "cpu"
    needed = sum(a.nbytes for a in data.values()) * 1.3 + (1 << 30)
    free, _total = th.cuda.mem_get_info()
    if needed > free:
        print(
            f"  dataset needs ~{needed / 1e9:.1f} GB on device but only "
            f"{free / 1e9:.1f} GB VRAM is free; cloning on cpu "
            f"(force with DEVICE=cuda)."
        )
        return "cpu"
    return "cuda"


def clone_expert_main(
    expert,
    deck_path,
    default_dataset,
    default_out,
    description,
    reward_module=None,
    set_deck=None,
    default_episodes=6000,
    argv=None,
):
    """Collect-then-clone entry point shared by the per-expert BC modules
    (training/bc_iono.py and friends).

    Lives here rather than being copied into each of them for the same reason
    bc_crustle.py imports `behavior_clone` instead of duplicating it: the
    passes must not drift on the dataset-cache invalidation rule, the CLI
    defaults, or which arguments reach behavior_clone. Only the expert, its
    deck, and the artifact names differ, and those are the parameters.

    Args:
        expert: callable(obs_dict) -> list[int], the heuristic to imitate.
        deck_path: the deck `expert` was written for; the learner seat pilots
            it (see collect_expert_dataset).
        default_dataset / default_out: per-expert artifact names. These MUST
            be distinct per expert -- every expert's dataset has an identical
            obs width, so a shared cache filename would sail straight past the
            width check below and clone the wrong expert.
        set_deck: optional callable(list[int]) pinning the expert's own cached
            deck to the one the learner plays, so the two cannot drift if
            either file is edited. Agents written as standalone Kaggle
            submissions (lucario) don't expose one.
        default_episodes: --episodes default. Per-expert because episodes are
            a poor proxy for dataset size -- samples per game vary by more
            than 2x across these decks -- and it is SAMPLES that hit the
            memory wall (see the --episodes comment below).
    """
    parser = argparse.ArgumentParser(description=description)
    # Collection runs ~30 games/s single-process. More data is the main lever
    # against the val plateau (train loss kept falling while val went flat).
    #
    # But there is a ceiling, and it is on samples rather than episodes. Each
    # sample is an OBS_SIZE float32 row, and collection holds the whole set
    # twice at the end -- once as the per-step list, once as the array built
    # from it -- so N samples peak at ~2 * N * OBS_SIZE * 4 bytes. At
    # OBS_SIZE=3312 that is ~26 KB per sample: 875k samples (iono at 6000
    # episodes) peaks near 22 GiB and dies on a 31 GiB box, and it dies AFTER
    # paying for the whole collection. Decks differ enough here to matter --
    # iono averages ~146 samples/game against ~75 for the others -- so size
    # default_episodes per expert against the sample count you want, and keep
    # roughly 500k as the comfortable ceiling for a 32 GiB machine.
    parser.add_argument(
        "--episodes", type=int, default=default_episodes, help="games to collect"
    )
    # Epochs is an upper bound: training early-stops on val plateau and saves
    # the best-val checkpoint, so a high cap just gives it room.
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--patience", type=int, default=8, help="early-stop patience (epochs)"
    )
    parser.add_argument(
        "--dataset",
        default=default_dataset,
        help="dataset cache; loaded if it exists (delete to recollect)",
    )
    parser.add_argument("--out", default=default_out, help="model save path")
    # Resolved after the dataset is known -- the choice depends on its size
    # (see _auto_device). Passing --device explicitly skips that entirely.
    parser.add_argument(
        "--device", default=None, help="cpu/cuda (default: cuda if it fits)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    data = None
    if os.path.exists(args.dataset):
        print(f"Loading cached dataset {args.dataset}")
        with np.load(args.dataset) as npz:
            data = {k: npz[k] for k in npz.files}
        # The observation layout can change (obs_vectorizer.py), so a cache
        # from an older width would feed wrong-sized vectors into the new
        # policy net. Detect the mismatch and re-collect rather than crash
        # cryptically.
        if data["obs"].shape[1] != CabtEnv.OBS_SIZE:
            print(
                f"  cached obs width {data['obs'].shape[1]} != current OBS_SIZE "
                f"{CabtEnv.OBS_SIZE} (vectorizer changed); re-collecting."
            )
            data = None

    if data is None:
        if set_deck is not None:
            set_deck(_load_deck(deck_path))
        print(f"Collecting {args.episodes} expert episodes on {deck_path}...")
        start = time.perf_counter()
        data = collect_expert_dataset(
            expert,
            args.episodes,
            seed=args.seed,
            deck_path=deck_path,
            reward_module=reward_module,
        )
        print(
            f"Collected {len(data['obs']):,} samples in "
            f"{time.perf_counter() - start:.0f}s; caching to {args.dataset}"
        )
        np.savez_compressed(args.dataset, **data)

    return behavior_clone(
        data,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=args.device or _auto_device(data),
        seed=args.seed,
    )


def main():
    # This file's own pass is now just clone_expert_main with the crustle
    # expert: same defaults, same behavior (deck_path=DECK_PATH and the
    # default crustle reward module are exactly what it used before), with the
    # CLI body shared instead of a third copy of it living in this file.
    clone_expert_main(
        crustle_agent,
        DECK_PATH,
        default_dataset="bc_dataset.npz",
        default_out="ppo_crustle_bc",
        description="Behavior-clone the crustle heuristic into a MaskablePPO "
        "policy (see module docstring).",
    )


if __name__ == "__main__":
    main()
