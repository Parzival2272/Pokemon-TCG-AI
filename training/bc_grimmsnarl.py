"""Behavior-cloning (BC) warm start from the grimmsnarl heuristic.

One of four per-expert counterparts to training/bc_crustle.py -- see
training/bc_iono.py for the shared rationale (why the collection loop, the
clone and the CLI all live in bc.py, why the learner seat pilots the expert's
own deck via CabtEnv's `deck_path` instead of cabt_env.DECK_PATH, and why the
value head is fit against the deck-agnostic training/outcome_rewards rather
than crustle_rewards).

Usage:
    python -m training.bc_grimmsnarl        # collect + clone with defaults
    BC_INIT=ppo_grimmsnarl_bc.zip python -m training.train

See bc_iono.py's usage note before warm-starting train.py from this: train.py
still plays cabt_env.DECK_PATH (the crustle list) against crustle_rewards.
"""

from heuristics.grimmsnarl_agent import agent as grimmsnarl_agent
from heuristics.grimmsnarl_agent import set_deck as grimmsnarl_set_deck
from training import outcome_rewards
from training.bc import clone_expert_main

DECK_PATH = "heuristics/grimmsnarl_agent/deck.csv"


def main():
    clone_expert_main(
        grimmsnarl_agent,
        DECK_PATH,
        # Per-expert cache name: every expert's dataset has an identical obs
        # width, so a shared filename would sail past bc.py's width check and
        # clone the wrong expert.
        default_dataset="bc_dataset_grimmsnarl.npz",
        default_out="ppo_grimmsnarl_bc",
        description="Behavior-clone the grimmsnarl heuristic into a "
        "MaskablePPO policy (see module docstring).",
        reward_module=outcome_rewards,
        set_deck=grimmsnarl_set_deck,
    )


if __name__ == "__main__":
    main()
