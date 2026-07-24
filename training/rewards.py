"""
Rewards are always from the perspective of the player who took the step's
action. In heuristic-opponent mode that's
the fixed learner side; in self-play it's whichever player was to move
(prev_obs's yourIndex), since the reward is paired with prev_obs in the
rollout.

The base signal is the sparse terminal +1 win / -1 loss. Because a game spans
hundreds of decisions, that alone is a very thin gradient, so on non-terminal
steps we add a dense prize-differential term: reward for each prize the current learner takes,
penalty for each prize the opponent takes since the previous decision.

Prize mechanic: you take from your OWN prize pile
when you knock out an opponent's Pokemon, so a player's remaining prize count
*decreases* as they win, reaching 0 on a prize-out win. Hence "prizes taken" is
prev_remaining - now_remaining. Counts are read by absolute player index, so the
term is independent of whose turn the observation reflects. Each side's "taken"
is clamped to >= 0 so the setup-time jump from 0 to 6 prizes (and rare
prize-returning effects) doesn't register as a swing.
"""

# Weight of one prize swing.
PRIZE_REWARD = 0.1


def _prizes_remaining(obs_dict, player_index):
    players = (obs_dict.get("current") or {}).get("players") or [{}, {}]
    if player_index < len(players):
        return len(players[player_index].get("prize") or [])
    return 0


def compute_reward(prev_obs, cur_obs, done, result, me_index):
    """Reward from the acting player's perspective.

    Args:
        prev_obs: the observation the action was chosen from.
        cur_obs: observation after this action.
        done: True if the battle ended on this step.
        result: winning player index (== me_index means "me" won).
        me_index: absolute index (0/1) of the acting player -- the fixed
            learner vs a heuristic opponent, or the mover in self-play.

    Returns:
        float: +1/-1 on a terminal step, otherwise the prize-differential
        shaping term.
    """
    if done:
        return 1.0 if result == me_index else -1.0

    opp_index = 1 - me_index
    my_took = max(
        0,
        _prizes_remaining(prev_obs, me_index) - _prizes_remaining(cur_obs, me_index),
    )
    opp_took = max(
        0,
        _prizes_remaining(prev_obs, opp_index) - _prizes_remaining(cur_obs, opp_index),
    )
    return PRIZE_REWARD * (my_took - opp_took)
