"""Behavior-cloning (BC) warm start from the alakazam_v2 heuristic.

One of four per-expert counterparts to training/bc_crustle.py -- see
training/bc_iono.py for the shared rationale (why the collection loop, the
clone and the CLI all live in bc.py, why the learner seat pilots the expert's
own deck via CabtEnv's `deck_path` instead of cabt_env.DECK_PATH, and why the
value head is fit against the deck-agnostic training/outcome_rewards rather
than crustle_rewards).

One thing is specific to this expert. alakazam_v2 ships a 2-ply minimax
(ptcg.api search_begin/step/end) layered on top of its heuristic, budgeted at
0.8s per MAIN decision, and it defaults to ON in the agent module. CabtEnv
does supply search_begin_input, so it really runs here -- train.py measured
~27x on rollout wall-clock with it enabled, and that was with alakazam_v2 as
one of eleven *opponents*. As the expert it drives EVERY learner decision, so
6000 episodes of collection would be a multi-day run instead of minutes.

train.py already handles this: it sets the module flag from the
ALAKAZAM_V2_SEARCH env var at import time (default off), and
collect_expert_dataset imports train.py for OPPONENT_POOL, so that assignment
lands before any episode is rolled here too. Be aware of what that means for
the clone: by default this imitates alakazam_v2's *heuristic core* -- the
tuned weights, the Hammer-aware lethal search, the Teleportation guard -- and
not its minimax. That is the intended target; the minimax's picks are a
function of a search tree the policy cannot see, so they are the part of this
agent a feed-forward net has least hope of reproducing. Set
ALAKAZAM_V2_SEARCH=1 to clone the full agent if you are willing to pay for it.

Usage:
    python -m training.bc_alakazam_v2       # collect + clone with defaults
    BC_INIT=ppo_alakazam_v2_bc.zip python -m training.train

See bc_iono.py's usage note before warm-starting train.py from this: train.py
still plays cabt_env.DECK_PATH (the crustle list) against crustle_rewards.
"""

from heuristics.alakazam_v2_agent import agent as alakazam_v2_agent
from heuristics.alakazam_v2_agent import set_deck as alakazam_v2_set_deck
from training import outcome_rewards
from training.bc import clone_expert_main

DECK_PATH = "heuristics/alakazam_v2_agent/deck.csv"


def main():
    clone_expert_main(
        alakazam_v2_agent,
        DECK_PATH,
        # Per-expert cache name: every expert's dataset has an identical obs
        # width, so a shared filename would sail past bc.py's width check and
        # clone the wrong expert.
        default_dataset="bc_dataset_alakazam_v2.npz",
        default_out="ppo_alakazam_v2_bc",
        description="Behavior-clone the alakazam_v2 heuristic into a "
        "MaskablePPO policy (see module docstring).",
        reward_module=outcome_rewards,
        set_deck=alakazam_v2_set_deck,
    )


if __name__ == "__main__":
    main()
