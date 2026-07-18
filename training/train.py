import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from training.cabt_env import CabtEnv
from training.callbacks import WinRateCallback

from heuristics.crustle_agent import agent as crustle_agent
from heuristics.abomasnow_agent import agent as abomasnow_agent
from heuristics.dragapult_agent import agent as dragapult_agent
from heuristics.dragapult_v2_agent import agent as dragapult_v2_agent
from heuristics.iono_agent import agent as iono_agent
from heuristics.archaludon_agent import agent as archaludon_agent
from heuristics.ragingbolt_agent import agent as ragingbolt_agent
from heuristics.alakazam_agent import agent as alakazam_agent
from heuristics.starmie_agent import agent as starmie_agent

# ragingbolt_agent ships with a per-step hand/option dump behind DEBUG; silence
# it so a training run with hundreds of thousands of steps isn't flooded.
import heuristics.ragingbolt_agent.ragingbolt_agent as _ragingbolt_module

_ragingbolt_module.DEBUG = False


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
    (
        "ragingbolt",
        ragingbolt_agent,
        _load_deck("heuristics/ragingbolt_agent/deck.csv"),
    ),
    ("alakazam", alakazam_agent, _load_deck("heuristics/alakazam_agent/deck.csv")),
    ("starmie", starmie_agent, _load_deck("heuristics/starmie_agent/deck.csv")),
]

# Each worker runs the native game engine in its own OS process, since
# Battle.battle_ptr in ptcg/sim.py is global mutable state shared within a
# process -- multiple envs in one process would clobber each other's battle.
# Worker count is a CPU-bound decision (each worker is one OS process
# stepping the native engine), not a GPU one -- it scales with cores on
# whatever machine this runs on, leaving a couple of cores free for the OS
# and the main training process. Override via the N_ENVS env var if you want
# a fixed count instead.
N_ENVS = int(os.environ.get("N_ENVS", max(1, (os.cpu_count() or 4) - 2)))

# Split workers between pure self-play (free exploration, both sides RL) and
# a heuristic opponent (directly optimizes for beating the known baselines --
# a random one from OPPONENT_POOL each episode). Must sum to N_ENVS.
N_SELFPLAY_ENVS = N_ENVS // 6
N_HEURISTIC_ENVS = N_ENVS - N_SELFPLAY_ENVS

# Keep total samples collected per policy update roughly constant regardless
# of N_ENVS, rather than letting it balloon (or shrink) with worker count.
TARGET_SAMPLES_PER_UPDATE = 2048
N_STEPS = max(TARGET_SAMPLES_PER_UPDATE // N_ENVS, 1)


def mask_fn(env):
    return env.action_masks()


def make_selfplay_env():
    env = CabtEnv()
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
    env_fns = [make_selfplay_env] * N_SELFPLAY_ENVS + [
        make_heuristic_env
    ] * N_HEURISTIC_ENVS
    env = SubprocVecEnv(env_fns)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=N_STEPS,  # N_STEPS * N_ENVS ~= TARGET_SAMPLES_PER_UPDATE
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./ppo_cabt_logs/",
    )

    model.learn(total_timesteps=100_000, callback=WinRateCallback())
    model.save("ppo_crustle")
