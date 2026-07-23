"""Reward shaping for the RL learner (used by CabtEnv in both modes).

Rewards are always from the perspective of the player who took the step's
action -- "me" (an absolute player index). In heuristic-opponent mode that's
the fixed learner side; in self-play it's whichever player was to move
(prev_obs's yourIndex), since the reward is paired with prev_obs in the
rollout.

The base signal is the sparse terminal +1 win / -1 loss. Because a game spans
hundreds of decisions, that alone is a very thin gradient, so on non-terminal
steps we add a dense prize-differential term: reward for each prize "me" takes,
penalty for each prize the opponent takes since the previous decision. On top
of that there's a small bonus for taking multiple prizes in the same step
(a multi-knockout swing), a slight bonus for evolving a Pokemon, a very
small bonus for attaching an energy card, and a small bonus for damage "me"
deals to any of the opponent's Pokemon -- active or bench.

Prize mechanic (verified against the engine): you take from your OWN prize pile
when you knock out an opponent's Pokemon, so a player's remaining prize count
*decreases* as they win, reaching 0 on a prize-out win. Hence "prizes taken" is
prev_remaining - now_remaining. Counts are read by absolute player index, so the
term is independent of whose turn the observation reflects. Each side's "taken"
is clamped to >= 0 so the setup-time jump from 0 to 6 prizes (and rare
prize-returning effects) doesn't register as a swing.
"""

# Weight of one prize swing. Kept well under 1.0 so the terminal win/loss
# stays the dominant signal: at most 6 prizes -> +/-0.6 of shaping vs the
# +/-1.0 outcome reward.
PRIZE_REWARD = 0.1

# Extra multiplier on my_took's prize reward when 2+ prizes are taken in a
# single step (e.g. a single attack knocking out a multi-prize Pokemon, or
# two knockouts before the next decision point). Rewards the tempo/value of
# a multi-prize turn without letting it swamp the base per-prize signal:
# two prizes -> 0.25 instead of the linear 0.2.
MULTI_PRIZE_MULTIPLIER = 1.25

# Small bonus per evolution "me" performs since the previous decision. Kept
# far below PRIZE_REWARD since evolving is a minor developmental move, not a
# decisive event like a knockout.
EVOLVE_REWARD = 0.02

# Very small bonus per energy card "me" attaches since the previous decision.
# Smaller than EVOLVE_REWARD -- this just nudges the agent not to waste its
# once-per-turn energy attachment, it isn't meant to carry much signal.
ENERGY_ATTACH_REWARD = 0.01

# Reward for damage "me" deals to any of the opponent's Pokemon (active or
# bench) since the previous decision: 0.01 per 100 damage. Rewards chip
# damage/setup toward a knockout even on steps that don't land the KO itself.
DAMAGE_REWARD_PER_100 = 0.01


def _prizes_remaining(obs_dict, player_index):
    players = (obs_dict.get("current") or {}).get("players") or [{}, {}]
    if player_index < len(players):
        return len(players[player_index].get("prize") or [])
    return 0


def _evolve_count(obs_dict, player_index):
    """Count Evolve events attributed to player_index since the previous decision.

    NOTE: unlike the prize mechanic above, this is NOT verified against a live
    observation -- I inferred the shape from strings in the compiled engine
    (kaggle_environments' cabt/cg/libcg.so is a Linux binary I can't load or
    run here to confirm). It assumes obs_dict["logs"] is a list of event dicts
    including entries like {"type": "Evolve", "playerIndex": <int>, ...},
    since "Evolve" appears among log-type-looking constants (alongside
    "Retreat"/"Ability"/"Discard"/"Attach") and "playerIndex" appears among
    per-event target fields in the binary's string table. Please confirm the
    real field names against a printed obs["logs"] from an actual game (e.g.
    trigger an evolution and inspect cur_obs["logs"]) and adjust this function
    if they differ -- as written it fails safe (returns 0, no crash) rather
    than raising if the schema doesn't match.
    """
    count = 0
    for entry in obs_dict.get("logs") or []:
        if (
            isinstance(entry, dict)
            and entry.get("type") == "Evolve"
            and entry.get("playerIndex") == player_index
        ):
            count += 1
    return count


def _energy_attach_count(obs_dict, player_index):
    """Count energy-Attach events by player_index since the previous decision.

    Same caveat as _evolve_count: inferred from the compiled engine's string
    table, not verified against a live observation. "Attach" appears as a
    log-type constant alongside "Evolve"/"Ability"/"Discard"/"Retreat", and
    the string table separately lists both "energyIndex" and "toolIndex" as
    per-event fields -- consistent with a single "Attach" type covering both
    energy and tool attachment, disambiguated by which index field is set.
    This only counts entries that look like an energy attach (energyIndex
    present). Confirm against a real obs["logs"] and adjust if tool attaches
    also set energyIndex, or if energy attaches turn out to use a different
    type/field; as written it fails safe (returns 0) rather than raising if
    the schema doesn't match.
    """
    count = 0
    for entry in obs_dict.get("logs") or []:
        if (
            isinstance(entry, dict)
            and entry.get("type") == "Attach"
            and entry.get("playerIndex") == player_index
            and entry.get("energyIndex") is not None
        ):
            count += 1
    return count


def _damage_dealt(obs_dict, target_player_index):
    """Sum HP lost by target_player_index's Pokemon (active + bench) since the
    previous decision, from HpChange log entries.

    Same caveat as the other _*_count helpers: inferred from the compiled
    engine's string table, not verified against a live observation. Assumes
    obs_dict["logs"] entries look like {"type": "HpChange",
    "playerIndex": <whose Pokemon changed>, "value": <magnitude>,
    "isRecover": <bool>, ...}, since "HpChange", "value", and "isRecover" all
    appear together in the string table. Deliberately doesn't filter by
    inPlayArea so bench damage (splash/spread attacks) counts same as active
    damage. Excludes entries where isRecover is true (healing). Confirm the
    field names/signs against a real obs["logs"] and adjust if they differ;
    fails safe (contributes 0) rather than raising on a schema mismatch.
    """
    total = 0
    for entry in obs_dict.get("logs") or []:
        if not (
            isinstance(entry, dict)
            and entry.get("type") == "HpChange"
            and entry.get("playerIndex") == target_player_index
            and not entry.get("isRecover")
        ):
            continue
        value = entry.get("value")
        if isinstance(value, (int, float)) and value > 0:
            total += value
    return total


def compute_reward(prev_obs, cur_obs, done, result, me_index):
    """Reward from the acting player's perspective.

    Args:
        prev_obs: the observation the action was chosen from (non-terminal).
        cur_obs: observation after this action (and any opponent auto-play).
        done: True if the battle ended on this step.
        result: winning player index (== me_index means "me" won).
        me_index: absolute index (0/1) of the acting player -- the fixed
            learner vs a heuristic opponent, or the mover in self-play.

    Returns:
        float: +1/-1 on a terminal step, otherwise the prize-differential
        shaping term (with a multi-prize bonus) plus the evolve,
        energy-attach, and damage bonuses.
    """
    if done:
        return 1.0 if result == me_index else -1.0

    opp_index = 1 - me_index
    my_took = max(
        0,
        _prizes_remaining(prev_obs, me_index)
        - _prizes_remaining(cur_obs, me_index),
    )
    opp_took = max(
        0,
        _prizes_remaining(prev_obs, opp_index)
        - _prizes_remaining(cur_obs, opp_index),
    )

    my_prize_reward = PRIZE_REWARD * my_took
    if my_took >= 2:
        my_prize_reward *= MULTI_PRIZE_MULTIPLIER
    opp_prize_reward = PRIZE_REWARD * opp_took

    evolve_reward = EVOLVE_REWARD * _evolve_count(cur_obs, me_index)
    energy_reward = ENERGY_ATTACH_REWARD * _energy_attach_count(cur_obs, me_index)
    damage_reward = DAMAGE_REWARD_PER_100 * (_damage_dealt(cur_obs, opp_index) / 100)

    return (
        my_prize_reward
        - opp_prize_reward
        + evolve_reward
        + energy_reward
        + damage_reward
    )
