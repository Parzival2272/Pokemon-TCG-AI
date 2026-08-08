"""Behavior-cloning (BC) warm start from the iono heuristic.

One of four per-expert counterparts to training/bc_crustle.py -- alongside
bc_alakazam_v2.py, bc_grimmsnarl.py and bc_lucario.py. All of them are this
thin on purpose: the collection loop (`collect_expert_dataset`), the clone
(`behavior_clone`) and the CLI (`clone_expert_main`) all live in bc.py, so the
passes cannot drift on net_arch, the best-val checkpointing, the value-head
fit, or the dataset-cache invalidation rule. Only the four constants below --
expert, deck, dataset cache, output name -- differ.

How this differs from bc_crustle.py: that file clones the expert that pilots
cabt_env.DECK_PATH, the deck train.py's learner plays, so it asserts DECK_PATH
really is the crustle list and takes the env's default (crustle) shaping. The
iono expert pilots a different deck, so instead:

  * DECK_PATH is not consulted at all. CabtEnv takes `deck_path` and the
    learner seat pilots heuristics/iono_agent/deck.csv -- the list the iono
    heuristic was written for. (Opponents are the usual OPPONENT_POOL and
    pilot their own decks, exactly as in the crustle pass; iono appears in
    that pool too, making some episodes a mirror match.)

  * The value head is fit against training/outcome_rewards, not
    crustle_rewards. Nearly every crustle shaping term keys on the crustle
    list's card IDs, and several are penalties grading a board against how the
    crustle deck should look -- pointed at the iono list they would score its
    normal play as malformed crustle play. outcome_rewards keeps only what is
    true for any deck: the +-1 game result plus prize differential.

Usage:
    python -m training.bc_iono              # collect + clone with defaults
    BC_INIT=ppo_iono_bc.zip python -m training.train

The saved zip is a normal MaskablePPO save with the same POLICY_NET_ARCH every
other pass uses, so train.py's BC_INIT env var can warm-start from it -- but
note train.py's learner still plays cabt_env.DECK_PATH (the crustle list) and
trains against crustle_rewards. To actually train this deck, point those at
the iono list first; warm-starting a crustle run from an iono clone would
start it off imitating lines its deck cannot play.
"""

from heuristics.iono_agent import agent as iono_agent
from heuristics.iono_agent import set_deck as iono_set_deck
from training import outcome_rewards
from training.bc import clone_expert_main

DECK_PATH = "heuristics/iono_agent/deck.csv"


def main():
    clone_expert_main(
        iono_agent,
        DECK_PATH,
        # Deliberately not bc.py's "bc_dataset.npz" or any other pass's cache:
        # every expert's dataset has an identical obs width, so a shared
        # filename would sail past bc.py's width check and clone the wrong
        # expert.
        default_dataset="bc_dataset_iono.npz",
        default_out="ppo_iono_bc",
        description="Behavior-clone the iono heuristic into a MaskablePPO "
        "policy (see module docstring).",
        reward_module=outcome_rewards,
        set_deck=iono_set_deck,
        # Below the other passes' 6000 because this deck is far wordier per
        # game: measured at ~146 samples/game here against ~75 for
        # grimmsnarl/lucario/alakazam_v2, since Iono/Bellibolt chains many
        # more individually-selected item and search effects per turn. 6000
        # episodes collected 875k samples and then died allocating the final
        # array (~22 GiB peak on a 31 GiB box) -- see the --episodes comment
        # in bc.py. 3000 lands at ~440k, in line with the other three, so the
        # clone is trained on a comparable dataset rather than a bigger one.
        default_episodes=3000,
    )


if __name__ == "__main__":
    main()
