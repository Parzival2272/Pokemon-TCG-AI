"""Deck-agnostic reward module for CabtEnv.

Same interface as training/crustle_rewards.py -- `reset_turn_tracking()` plus
`reward_terms(prev_obs, cur_obs, done, result, me_index)` -- so it can be
handed to `CabtEnv(reward_module=...)` in place of it.

Why it exists: crustle_rewards.py is a large, tuned, *deck-specific* shaping
function. Nearly every term keys on the crustle list's card IDs (Dwebble /
Crustle, Mega Kangaskhan ex, Hero's Cape, Hilda, Petrel...), and several are
penalties -- `board_shape`, `kangaskhan_bench`, `retreat`, `run_errand` --
that grade a board against how the crustle deck is supposed to look. Pointed
at the iono / alakazam_v2 / grimmsnarl / lucario lists it does not merely go
quiet: it scores those decks' normal play as malformed crustle play.

The behavior-cloning passes for those four decks (training/bc_iono.py and
friends) only need rewards for one thing -- the discounted return-to-go that
`behavior_clone` fits the value head to. So this module gives the part of the
signal that is true for any deck and nothing else:

    terminal    +-1.0 on the deciding step (the actual objective)
    prize_mine  +0.1 per prize the acting player just took
    prize_opp   -0.1 per prize their opponent just took

Prize count is the game's own scoreboard -- six to zero, no deck reads it
differently -- which makes it the one dense term that transfers unchanged.
Deliberately NOT included: damage, evolution, energy attachment, draw and
supporter/stadium terms. Those are only "good" relative to a game plan, and
guessing at a plan we do not have is what this module exists to avoid.

Stateless: unlike crustle_rewards, no term depends on turn counters or
one-shot latches, so `reset_turn_tracking()` is a no-op kept for interface
compatibility.
"""

# Matches crustle_rewards.PRIZE_REWARD so a return-to-go computed here is on
# the same scale as one from the crustle pass -- BC hyperparameters (the value
# loss's vf_coef against the policy NLL) then carry over without a re-tune.
PRIZE_REWARD = 0.1


def reset_turn_tracking():
    """No-op: this module keeps no per-game state (see module docstring).

    Present so CabtEnv can call it unconditionally on every reset, whichever
    reward module it was built with.
    """


# Alias matching crustle_rewards.reset_game_state.
reset_game_state = reset_turn_tracking


def _prizes_remaining(obs_dict, player_index):
    players = (obs_dict.get("current") or {}).get("players") or [{}, {}]
    if player_index >= len(players):
        return 0
    return len((players[player_index] or {}).get("prize") or [])


def reward_terms(prev_obs, cur_obs, done, result, me_index):
    """Named breakdown of the reward for the player who just acted.

    Args/returns match crustle_rewards.reward_terms. Unlike that one this is
    side-effect free, but call it once per step anyway so the scalar reward
    and the logged breakdown stay the same number.
    """
    if done:
        return {"terminal": 1.0 if result == me_index else -1.0}

    opp_index = 1 - me_index
    # max(0, ...) because prizes only ever come off the top: a negative delta
    # would mean the count grew, which no card does, so clamp rather than pay
    # out on an obs quirk.
    my_took = max(
        0, _prizes_remaining(prev_obs, me_index) - _prizes_remaining(cur_obs, me_index)
    )
    opp_took = max(
        0,
        _prizes_remaining(prev_obs, opp_index) - _prizes_remaining(cur_obs, opp_index),
    )
    return {
        "prize_mine": PRIZE_REWARD * my_took,
        "prize_opp": -PRIZE_REWARD * opp_took,
    }


def compute_reward(prev_obs, cur_obs, done, result, me_index):
    """Total reward -- the sum of reward_terms() (see it for the arguments)."""
    return sum(reward_terms(prev_obs, cur_obs, done, result, me_index).values())
