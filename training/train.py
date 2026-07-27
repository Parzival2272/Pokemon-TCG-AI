import os
import time
from datetime import timedelta

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from training.cabt_env import DECK_PATH, POLICY_NET_ARCH, CabtEnv
from training.callbacks import RewardTermCallback, SnapshotCallback, WinRateCallback
from training.league import SnapshotOpponentPool

from heuristics.crustle_agent import agent as crustle_agent
from heuristics.abomasnow_agent import agent as abomasnow_agent
from heuristics.dragapult_agent import agent as dragapult_agent
from heuristics.dragapult_v2_agent import agent as dragapult_v2_agent
from heuristics.iono_agent import agent as iono_agent
from heuristics.archaludon_agent import agent as archaludon_agent
from heuristics.alakazam_agent import agent as alakazam_agent
from heuristics.starmie_agent import agent as starmie_agent


def _load_deck(path):
    with open(path) as f:
        deck = [int(x) for x in f.read().splitlines() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{path} must contain 60 cards, got {len(deck)}")
    return deck


# Heuristic opponent pool: one is drawn at random each episode (see CabtEnv).
# Each entry is (name, agent_fn, deck) and the opponent pilots its OWN deck --
# a heuristic piloting a foreign deck wouldn't exercise the strategy it was
# written for. The learner always plays the CabtEnv DECK_PATH deck.
OPPONENT_POOL = [
    ("crustle", crustle_agent, _load_deck("heuristics/crustle_agent/crustle_deck.csv")),
    ("abomasnow", abomasnow_agent, _load_deck("heuristics/abomasnow_agent/deck.csv")),
    ("dragapult", dragapult_agent, _load_deck("heuristics/dragapult_agent/deck.csv")),
    (
        "dragapult_v2",
        dragapult_v2_agent,
        _load_deck("heuristics/dragapult_v2_agent/deck.csv"),
    ),
    ("iono", iono_agent, _load_deck("heuristics/iono_agent/deck.csv")),
    (
        "archaludon",
        archaludon_agent,
        _load_deck("heuristics/archaludon_agent/deck.csv"),
    ),
    # ragingbolt is excluded for now: its discard logic is buggy (returns 2
    # picks on "discard exactly 3", IndexError on some board states), so it
    # plays artificially weak and inflates win rates. Re-add once fixed.
    ("alakazam", alakazam_agent, _load_deck("heuristics/alakazam_agent/deck.csv")),
    ("starmie", starmie_agent, _load_deck("heuristics/starmie_agent/deck.csv")),
]

# Which device MaskablePPO trains the policy/value networks on. Defaults to
# "cpu" -- measured on this repo (see the DEVICE=cuda vs DEVICE=cpu smoke
# test in conversation/PR history), "cuda" is 8x+ SLOWER here despite a real
# GPU being available, because rollout collection calls the policy once per
# env step (thousands of tiny sequential forward passes to pick each
# action), and per-call CUDA kernel-launch/sync overhead (especially bad
# under Windows' WDDM driver model) dwarfs the actual compute for a network
# this small (MlpPolicy's default 64x64 layers). GPU would only help if the
# network were much bigger or rollout collection were batched, neither of
# which is true here. Override via the DEVICE env var, e.g. `DEVICE=cuda
# python -m training.train`, if that ever changes -- N_ENVS/BATCH_SIZE/
# N_EPOCHS below scale themselves to whichever device is chosen. Only
# affects where the neural net does its forward/backward passes -- the
# N_ENVS game-engine workers below always run on CPU regardless.
DEVICE = os.environ.get("DEVICE", "cpu")
if DEVICE == "cuda" and not torch.cuda.is_available():
    print("DEVICE=cuda requested but no CUDA GPU is available; falling back to cpu.")
    DEVICE = "cpu"

# Two hyperparameter profiles, selected by DEVICE. CPU is this repo's
# original, proven configuration -- unchanged. CUDA is a separate profile
# for when the policy/value networks grow enough (more/wider layers, a
# CNN/LSTM feature extractor, etc.) for a GPU to actually win: bigger
# batch_size and n_epochs so a GPU update pass gets large, dense matmuls,
# one less CPU core reserved for the game-engine workers since the main
# process's own compute moves to the GPU, and policy_kwargs as the hook for
# a larger net_arch once the model itself grows. It is NOT yet a proven win
# -- see the DEVICE comment above for the measured 8x+ slowdown on today's
# tiny MlpPolicy -- this profile exists so switching DEVICE="cuda" is a
# single flag flip once the model is heavy enough to justify it, instead of
# a re-tune at that point.
_CPU_PROFILE = dict(reserved_cores=2, batch_size=256, n_epochs=4, policy_kwargs=None)
_GPU_PROFILE = dict(
    reserved_cores=1,
    batch_size=2048,
    n_epochs=15,
    policy_kwargs=None,  # e.g. dict(net_arch=[256, 256]) once the model grows
)
_profile = _GPU_PROFILE if DEVICE == "cuda" else _CPU_PROFILE

# Each worker runs the native game engine in its own OS process, since
# Battle.battle_ptr in ptcg/sim.py is global mutable state shared within a
# process -- multiple envs in one process would clobber each other's battle.
# Worker count is a CPU-bound decision (each worker is one OS process
# stepping the native engine) -- it scales with cores on whatever machine
# this runs on, minus reserved_cores (see profiles above) for the OS and
# (on CPU) the main process's own policy forward/backward passes. Override
# via the N_ENVS env var if you want a fixed count instead.
N_ENVS = int(
    os.environ.get("N_ENVS", max(1, (os.cpu_count() or 4) - _profile["reserved_cores"]))
)

# League self-play: SnapshotCallback freezes the live policy into
# SNAPSHOT_DIR every SNAPSHOT_FREQ timesteps (plus once at training start),
# keeping the newest MAX_SNAPSHOTS; league envs draw a random frozen snapshot
# each episode (mirror match on the learner's deck). Unlike the old pure
# self-play envs -- where both sides fed one rollout stream and GAE
# bootstrapped values across perspective flips -- every league transition is
# the learner's own, so the PPO update is correct, and playing recent past
# selves still gives self-play curriculum (win_rate/league ~50% is healthy).
SNAPSHOT_DIR = "./league_snapshots"
SNAPSHOT_FREQ = 100_000
MAX_SNAPSHOTS = 5

# Split workers between league self-play (vs frozen snapshots, see above) and
# a heuristic opponent (directly optimizes for beating the known baselines --
# a random one from OPPONENT_POOL each episode). Must sum to N_ENVS. League
# gets a bigger share than the old pure self-play split (1/6) since its
# gradients are now correct; tune if heuristic win rates stall.
N_LEAGUE_ENVS = N_ENVS // 4
N_HEURISTIC_ENVS = N_ENVS - N_LEAGUE_ENVS

# 2048 steps/env is the standard PPO rollout length, so total buffer size
# scales with N_ENVS instead of being held constant.
TARGET_SAMPLES_PER_UPDATE = 2048 * N_ENVS
N_STEPS = max(TARGET_SAMPLES_PER_UPDATE // N_ENVS, 1)

# Minibatch size for each gradient step, and passes over each rollout
# buffer per update -- both overridable via env vars regardless of profile.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", _profile["batch_size"]))
N_EPOCHS = int(os.environ.get("N_EPOCHS", _profile["n_epochs"]))
# Warm-starting from bc.py requires the PPO model to have the SAME architecture
# the BC model used (load_state_dict is strict), so default both profiles to
# the shared POLICY_NET_ARCH unless a profile explicitly pins its own kwargs.
POLICY_KWARGS = _profile["policy_kwargs"] or dict(net_arch=list(POLICY_NET_ARCH))

# Learning-rate schedule: start high so early updates move the policy off its
# random/BC init quickly, then decay to a floor so late training refines
# instead of thrashing (a constant 1e-4 does neither well over 30M steps).
# Decays linearly from LR_START to LR_END over the first LR_DECAY_FRAC of
# training, then holds LR_END flat for the remainder -- the "bottoming out".
LR_START = float(os.environ.get("LR_START", 3e-4))
LR_END = float(os.environ.get("LR_END", 3e-5))
LR_DECAY_FRAC = float(os.environ.get("LR_DECAY_FRAC", 0.8))


def linear_decay_to_floor(start: float, end: float, decay_frac: float):
    """SB3 schedule: takes progress_remaining (1.0 at start -> 0.0 at end)."""

    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining  # 0.0 -> 1.0 as training advances
        if decay_frac <= 0:
            return end
        frac = min(progress / decay_frac, 1.0)
        return start + frac * (end - start)

    return schedule


def mask_fn(env):
    return env.action_masks()


def make_league_env():
    # The pool is a callable, re-scanned each episode, so snapshots saved
    # mid-run join the league. Snapshots pilot the learner's own deck: a
    # mirror match, which is also what they were trained on. Until the first
    # snapshot lands (SB3 resets the envs before on_training_start fires, so
    # this worker's first episode always predates it) the pool falls back to
    # the heuristics rather than to pure self-play -- see SnapshotOpponentPool.
    env = CabtEnv(
        opponent_agents=SnapshotOpponentPool(
            SNAPSHOT_DIR, _load_deck(DECK_PATH), fallback=OPPONENT_POOL
        )
    )
    env = ActionMasker(env, mask_fn)
    # info_keywords lifts CabtEnv's per-episode "opponent" tag into
    # info["episode"] so WinRateCallback can read it.
    env = Monitor(env, info_keywords=("opponent",))
    return env


def make_heuristic_env():
    env = CabtEnv(opponent_agents=OPPONENT_POOL)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env, info_keywords=("opponent",))
    return env


if __name__ == "__main__":
    # Built FIRST, before the envs: its constructor clears SNAPSHOT_DIR so
    # this run's league is its own, and SubprocVecEnv workers glob that
    # directory from their very first reset (which SB3 performs before any
    # callback hook runs). Constructing it later would let those first
    # episodes draw the previous run's snapshots.
    snapshot_cb = SnapshotCallback(
        SNAPSHOT_DIR, SNAPSHOT_FREQ, max_snapshots=MAX_SNAPSHOTS, verbose=1
    )

    env_fns = [make_league_env] * N_LEAGUE_ENVS + [
        make_heuristic_env
    ] * N_HEURISTIC_ENVS
    env = SubprocVecEnv(env_fns)

    print(f"Training on device: {DEVICE}")
    print(
        f"LR schedule: {LR_START:.1e} -> {LR_END:.1e} over the first "
        f"{LR_DECAY_FRAC:.0%} of training, then flat"
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=DEVICE,
        learning_rate=linear_decay_to_floor(LR_START, LR_END, LR_DECAY_FRAC),
        n_steps=N_STEPS,  # N_STEPS * N_ENVS ~= TARGET_SAMPLES_PER_UPDATE
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        policy_kwargs=POLICY_KWARGS,
        gamma=0.995,
        # SB3's default is 0.0; a small entropy bonus keeps the policy
        # exploring instead of collapsing onto one action pattern early.
        ent_coef=0.01,
        tensorboard_log="./ppo_cabt_logs/",
    )

    # Warm start: BC_INIT=<path.zip> copies the policy weights (actor AND
    # value head) out of a behavior-cloned model (training/bc.py) so PPO
    # starts from "imitates the starmie heuristic" instead of random. Only
    # the network weights are taken -- optimizer state and PPO hyperparams
    # stay fresh from the model built above. The initial league snapshot
    # (SnapshotCallback at training start) then captures the BC policy too.
    bc_init = os.environ.get("BC_INIT")
    if bc_init:
        from stable_baselines3.common.save_util import load_from_zip_file

        _, params, _ = load_from_zip_file(bc_init, device=DEVICE)
        model.policy.load_state_dict(params["policy"])
        print(f"Warm-started policy from {bc_init}")

    total_timesteps = 10_000_000
    win_rate_cb = WinRateCallback()  # snapshot_cb was built above, before the envs
    reward_term_cb = RewardTermCallback()

    start = time.perf_counter()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[win_rate_cb, reward_term_cb, snapshot_cb],
    )
    elapsed = time.perf_counter() - start

    model_name = "ppo_starmie_v14"
    model.save(model_name)

    # ---- End-of-training report -------------------------------------------
    steps_done = model.num_timesteps
    avg_fps = steps_done / elapsed if elapsed > 0 else float("nan")

    summary = win_rate_cb.summary()
    # "overall" now covers the heuristic opponents only, so it undercounts the
    # episodes actually played -- sum the per-opponent entries for the total.
    total_games = sum(g for name, (_, g) in summary.items() if name != "overall")

    print("\n" + "=" * 60)
    print("TRAINING REPORT")
    print("=" * 60)
    print(f"Device               : {DEVICE}")
    print(f"Parallel envs        : {N_ENVS}")
    print(
        f"Learning rate        : {LR_START:.1e} -> {LR_END:.1e} "
        f"(linear over first {LR_DECAY_FRAC:.0%}, then flat)"
    )
    print(f"Timesteps            : {steps_done:,} / {total_timesteps:,}")
    print(f"Wall-clock time      : {timedelta(seconds=round(elapsed))} ({elapsed:.1f}s)")
    print(f"Average FPS          : {avg_fps:,.0f} steps/s")
    print(f"Per-env FPS          : {avg_fps / N_ENVS:,.0f} steps/s")
    print(f"Episodes completed   : {total_games:,}")
    if elapsed > 0:
        print(f"Episodes/hour        : {total_games / elapsed * 3600:,.0f}")

    print("\nWin rate (career, cumulative over run):")
    # Overall first, then per-opponent sorted worst matchup first. "league"
    # appears among them but is excluded from overall (see WinRateCallback);
    # it is a mirror match, so ~50% is the healthy reading, not a matchup score.
    overall = summary.pop("overall", None)
    if overall is not None:
        w, g = overall
        label = "overall (vs heuristics)"
        print(f"  {label:<22}: {w / g:6.1%}  ({w:,}/{g:,})" if g else f"  {label}: n/a")
    for name, (w, g) in sorted(summary.items(), key=lambda kv: kv[1][0] / kv[1][1] if kv[1][1] else 0):
        if g:
            print(f"  {name:<22}: {w / g:6.1%}  ({w:,}/{g:,})")
    print("=" * 60)
    print(f"Saved model to {model_name}.zip")
