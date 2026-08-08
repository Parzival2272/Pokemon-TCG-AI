"""Behavior-cloning (BC) warm start from the lucario heuristic.

One of four per-expert counterparts to training/bc_crustle.py -- see
training/bc_iono.py for the shared rationale (why the collection loop, the
clone and the CLI all live in bc.py, why the learner seat pilots the expert's
own deck via CabtEnv's `deck_path` instead of cabt_env.DECK_PATH, and why the
value head is fit against the deck-agnostic training/outcome_rewards rather
than crustle_rewards).

Two things are specific to this expert, both because lucario/ is the one
heuristic laid out as a standalone Kaggle submission rather than an agent
package:

  * Importing it goes through heuristics/lucario/__init__.py, which aliases
    the `cg` package onto ptcg before importing agent.py -- see that file;
    the vendored cg/ ships libcg.so only and cannot load on Windows.

  * There is no `set_deck`. agent.py reads its deck once at import into a
    module-level DECK and only ever returns it to answer the engine's
    deck-submission call (`obs["select"] is None`), which CabtEnv never
    routes to an agent -- battle_start is handed the deck lists directly. So
    the expert's play is deck-global-free and nothing needs pinning; we pass
    the same heuristics/lucario/deck.csv the OPPONENT_POOL entry uses.
    (Note that directory also holds lucario_deck.csv, the human-readable
    source deck.csv is built from -- deck.csv is the one to load.)

Usage:
    python -m training.bc_lucario           # collect + clone with defaults
    BC_INIT=ppo_lucario_bc.zip python -m training.train

See bc_iono.py's usage note before warm-starting train.py from this: train.py
still plays cabt_env.DECK_PATH (the crustle list) against crustle_rewards.
"""

from heuristics.lucario import agent as lucario_agent
from training import outcome_rewards
from training.bc import clone_expert_main

DECK_PATH = "heuristics/lucario/deck.csv"


def main():
    clone_expert_main(
        lucario_agent,
        DECK_PATH,
        # Per-expert cache name: every expert's dataset has an identical obs
        # width, so a shared filename would sail past bc.py's width check and
        # clone the wrong expert.
        default_dataset="bc_dataset_lucario.npz",
        default_out="ppo_lucario_bc",
        description="Behavior-clone the lucario heuristic into a MaskablePPO "
        "policy (see module docstring).",
        reward_module=outcome_rewards,
        # No set_deck: see module docstring.
    )


if __name__ == "__main__":
    main()
