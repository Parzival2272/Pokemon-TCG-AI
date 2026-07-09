import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from crustle_agent import agent as crustle_agent_fn
from training.cabt_env import CabtEnv

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
# a fixed heuristic opponent (directly optimizes for beating the known
# baseline). Must sum to N_ENVS.
N_SELFPLAY_ENVS = N_ENVS // 2
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
    env = Monitor(env)
    return env


def make_heuristic_env():
    env = CabtEnv(opponent_agent=crustle_agent_fn)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    env_fns = [make_selfplay_env] * N_SELFPLAY_ENVS + [make_heuristic_env] * N_HEURISTIC_ENVS
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

    model.learn(total_timesteps=1_000_00)
    model.save("ppo_crustle")
