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

On top of the original terms above, this file also adds a batch of
deck-specific shaping terms for the Mega Starmie ex / Mega Froslass ex list
(see starmie_agent.py for the full deck/strategy writeup this leans on):
card draw, Wally's Compassion healing, wasting an attached Ignition Energy on
a non-Nebula-Beam attack, landing Crushing Hammer, playing Items while the
opponent has Budew in play, Boss's Orders setting up a good KO, overfilling
the attacker bench, two matchup-specific evolution penalties, and failing to
attack on our 2nd turn or later. Several of these need card/attack IDs or
log-event names this file has no way to verify from here -- each helper below
documents its specific guess and fails safe (contributes 0) rather than
raising if the real schema differs. Confirm the flagged ones against real
obs["logs"]/obs["current"] dumps before trusting them in training.
"""

from collections import Counter


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

# Reward per card drawn since the previous decision (turn-start draw, Hilda,
# Lillie's Determination, or Dudunsparce's Run Away Draw all count the same
# way here -- see _draw_count).
DRAW_REWARD_PER_CARD = 0.005

# Wally's Compassion (heal all damage from one Mega): a fixed -0.12 charge
# plus 0.02 per 10 HP healed, breaking even around 60 HP healed and going
# negative below that -- Wally costs the turn's Supporter, so it should only
# come out ahead when it recovers real value (roughly a real attack's worth
# of damage or more), not when topped off after a single chip hit. Sized so
# this stays net-negative for a low-value heal even with SUPPORTER_PLAY_REWARD
# stacked on top (see _supporter_played_count).
WALLY_HEAL_BASE = -0.12
WALLY_HEAL_PER_10HP = 0.02

# Ignition Energy self-discards at end of turn, so attaching it and then
# attacking with anything other than Nebula Beam wastes it.
IGNITION_WASTE_PENALTY = 0.025

# Crushing Hammer actually landing (opponent loses an attached Energy, i.e.
# the coin flip hit) -- see _newly_discarded_ids.
HAMMER_DISCARD_REWARD = 0.02

# Small nudge to play Items while Budew is on the opponent's board.
ITEM_VS_BUDEW_REWARD = 0.005

# Boss's Orders pulling up a target our current board can actually punish:
# either a 2+ prize target Jetting Blow (120 dmg) can now KO, or a tankier
# target that needs Nebula Beam / a Mega Froslass ex attack (<=210) to finish.
BOSS_MULTI_PRIZE_REWARD = 0.03
BOSS_BIG_FINISH_REWARD = 0.03

# Bench plan calls for 2-3 attackers (Staryu/Snorunt), never more than 4;
# escalating penalty past that.
BENCH_OVER_4_PENALTY = 0.05
BENCH_OVER_5_PENALTY = 0.10

# Matchup-specific evolution penalties.
IONO_BELLIBOLT_EVOLVE_PENALTY = 0.5   # evolving into Mega Starmie ex vs Iono's Bellibolt ex
ARCHALUDON_EVOLVE_PENALTY = 0.5       # evolving into Mega Froslass ex vs Archaludon ex, still 3+ prizes down

# Ending our own turn (2nd turn onward) without having attacked.
NO_ATTACK_PENALTY = 0.05

# Resentful Refrain (50 x opponent's hand size) is weak when the opponent's
# hand is small -- penalize firing it for <=100 (i.e. opponent's hand <=2)
# on a step where Mega Starmie ex (Jetting Blow, generally the better play
# then) is also available to attack instead.
RESENTFUL_REFRAIN_LOW_VALUE_PENALTY = 0.025

# Bonus for choosing Absolute Snow (150 dmg + Sleep) into a target that
# Resentful Refrain's current damage (50 x opponent's hand) wouldn't have KO'd.
ABSOLUTE_SNOW_HIGH_HP_REWARD = 0.02

# Deck-out risk: reward using Run Away Draw (detected by Dudunsparce leaving
# play under its own ability) specifically when our deck is already thin --
# it's the one draw source in this deck that also replenishes deck size
# (shuffles itself + attachments back in), so it's the correct out here.
DECK_OUT_RISK_THRESHOLD = 10
RUN_AWAY_DRAW_DECK_SAVE_REWARD = 0.03

# Very small nudge to play the turn's Supporter at all -- including Wally's
# Compassion, since WALLY_HEAL_BASE's magnitude is sized to stay net-negative
# for a low-value heal even with this stacked on top.
SUPPORTER_PLAY_REWARD = 0.005

# Playing Risky Ruins; the bump bonus is additional, for specifically
# replacing an opposing Stadium with it (only one Stadium is in play at once).
STADIUM_PLAY_REWARD = 0.03
STADIUM_BUMP_REWARD = 0.05


# ── Card IDs (Staryu/Starmie/Snorunt/Froslass/Boss/Hammer/Ignition reused
# verbatim from starmie_agent.py's ids, already verified there against
# Card_ID_List_EN.pdf) plus optional ptcg.api access for everything this file
# can't get from raw numeric IDs alone (attack IDs, opponent-only card IDs,
# card types). Guarded the same way starmie_agent.py guards its `all_attack`
# import -- if ptcg.api isn't importable from wherever this module loads,
# every term below that depends on it just contributes 0 instead of raising.

STARYU_ID = 1030
STARMIE_ID = 1031      # Mega Starmie ex
SNORUNT_ID = 103        # Snorunt (TWM printing; 860 is the ASC alt)
FROSLASS_ID = 861       # Mega Froslass ex
BOSS_ID = 1182          # Boss's Orders
HAMMER_ID = 1120        # Crushing Hammer
IGNITION_ID = 17        # Ignition Energy
DUDUNSPARCE_ID = 66     # Dudunsparce (Run Away Draw)

try:
    from ptcg.api import all_card_data, CardType
    CARD_DB = {c.cardId: c for c in all_card_data()}
except Exception:
    CardType = None
    CARD_DB = {}


def _nth_attack_id(card_id, n):
    data = CARD_DB.get(card_id)
    atks = list(getattr(data, "attacks", None) or []) if data else []
    return atks[n] if n < len(atks) else None


def _find_card_ids_by_name(*name_fragments):
    """All CARD_DB ids whose name contains every fragment, case-insensitive
    (covers alt-art reprints sharing one name). NOT verified against a live
    observation or CARD_DB dump -- that CARD_DB entries even expose a `.name`
    attribute is itself an assumption, unlike every other field this file
    relies on. Print {CARD_DB[i].name for i in RESULT} to confirm the match
    is the intended card before trusting it. Empty set (fails safe) if
    CARD_DB is unavailable or nothing matches.
    """
    out = set()
    for cid, data in CARD_DB.items():
        name = (getattr(data, "name", "") or "").lower()
        if all(frag.lower() in name for frag in name_fragments):
            out.add(cid)
    return out


# Jetting Blow is Mega Starmie ex's printed 1st attack, Nebula Beam its 2nd --
# same order starmie_agent.py resolves them in. None (term disabled) if
# ptcg.api isn't importable here.
NEBULA_BEAM_ID = _nth_attack_id(STARMIE_ID, 1)

# Resentful Refrain is Mega Froslass ex's printed 1st attack, Absolute Snow
# its 2nd -- same resolution as NEBULA_BEAM_ID above.
RESENTFUL_REFRAIN_ID = _nth_attack_id(FROSLASS_ID, 0)
ABSOLUTE_SNOW_ID = _nth_attack_id(FROSLASS_ID, 1)

# Opponent-only cards with no verified numeric ID available to this file --
# resolved by name match instead. Confirm before relying on these.
BUDEW_IDS = _find_card_ids_by_name("budew")
IONO_BELLIBOLT_IDS = _find_card_ids_by_name("iono", "bellibolt")
ARCHALUDON_IDS = _find_card_ids_by_name("archaludon")


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


# ── Shared board/discard accessors ──

def _players(obs_dict):
    return (obs_dict.get("current") or {}).get("players") or [{}, {}]


def _pokemon_in_play(obs_dict, player_index):
    players = _players(obs_dict)
    if player_index >= len(players):
        return []
    p = players[player_index] or {}
    return [m for m in (list(p.get("active") or []) + list(p.get("bench") or [])) if m]


def _active_pokemon(obs_dict, player_index):
    players = _players(obs_dict)
    if player_index >= len(players):
        return None
    act = (players[player_index] or {}).get("active") or []
    return act[0] if act else None


def _count_in_play(obs_dict, player_index, card_id):
    return sum(1 for m in _pokemon_in_play(obs_dict, player_index) if m.get("id") == card_id)


def _opponent_has_any(obs_dict, card_ids, opp_index):
    if not card_ids:
        return False
    return any(m.get("id") in card_ids for m in _pokemon_in_play(obs_dict, opp_index))


def _newly_discarded_ids(prev_obs, cur_obs, player_index):
    """Card ids that entered player_index's discard pile since prev_obs, as a
    flat list (duplicates included). A count-based (not positional) diff, so
    it doesn't assume discard-pile ordering -- and it's a materially more
    solid way to detect "was card X played" than guessing a "Play" log-event
    schema would be, since it only relies on the `discard` zone list already
    used throughout starmie_agent.py (ps.discard). It will also fire on
    non-"play" discards (e.g. a cost paid to search with an Item), which is
    an accepted approximation, not a schema guess.
    """
    prev_players, cur_players = _players(prev_obs), _players(cur_obs)
    if player_index >= len(prev_players) or player_index >= len(cur_players):
        return []
    prev_ids = Counter(c.get("id") for c in ((prev_players[player_index] or {}).get("discard") or []) if c)
    cur_ids = Counter(c.get("id") for c in ((cur_players[player_index] or {}).get("discard") or []) if c)
    new_ids = []
    for cid, cnt in cur_ids.items():
        new_ids.extend([cid] * max(0, cnt - prev_ids.get(cid, 0)))
    return new_ids


def _is_energy_card(card_id):
    if not CARD_DB or CardType is None or card_id is None:
        return None
    data = CARD_DB.get(card_id)
    ct = getattr(data, "cardType", None) if data else None
    if ct is None:
        return None
    return ct in (getattr(CardType, "ENERGY", object()), getattr(CardType, "SPECIAL_ENERGY", object()))


def _is_item_card(card_id):
    if not CARD_DB or CardType is None or card_id is None:
        return None
    data = CARD_DB.get(card_id)
    ct = getattr(data, "cardType", None) if data else None
    return None if ct is None else ct == getattr(CardType, "ITEM", None)


def _prize_value_by_id(card_id):
    if not CARD_DB or card_id is None:
        return 1
    data = CARD_DB.get(card_id)
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


# ── New shaping-term helpers ──

def _draw_count(obs_dict, player_index):
    """Cards drawn by player_index since the previous decision, from Draw log
    entries. NOT verified against a live observation (same caveat tier as
    _evolve_count/_energy_attach_count above) -- "Draw" is a guess at the log
    type name, unconfirmed. If a single Draw entry represents multiple cards
    (e.g. Run Away Draw's "draw 3"), a "count"/"amount" field is summed if
    present, else each entry counts as 1 card. Fails safe (0) on a schema
    mismatch rather than raising.
    """
    total = 0
    for entry in obs_dict.get("logs") or []:
        if not (isinstance(entry, dict) and entry.get("type") == "Draw"
                and entry.get("playerIndex") == player_index):
            continue
        amount = entry.get("count", entry.get("amount"))
        total += amount if isinstance(amount, (int, float)) else 1
    return total


def _heal_dealt(obs_dict, player_index):
    """HP healed on player_index's own Pokemon since the previous decision --
    the mirror image of _damage_dealt's excluded isRecover branch, same
    schema caveats apply."""
    total = 0
    for entry in obs_dict.get("logs") or []:
        if not (isinstance(entry, dict) and entry.get("type") == "HpChange"
                and entry.get("playerIndex") == player_index
                and entry.get("isRecover")):
            continue
        value = entry.get("value")
        if isinstance(value, (int, float)) and value > 0:
            total += value
    return total


def _active_has_ignition(obs_dict, player_index):
    mon = _active_pokemon(obs_dict, player_index)
    if not mon:
        return False
    return any(isinstance(c, dict) and c.get("id") == IGNITION_ID
               for c in (mon.get("energyCards") or []))


def _attack_id_used(obs_dict, player_index):
    """attackId of an Attack log entry taken by player_index this step, if
    any (None otherwise). "attackId"/"playerIndex" on Attack entries are
    trusted -- starmie_agent.py already reads LogType.ATTACK's .attackId
    directly (its Itchy Pollen check). The "Attack" string spelling of that
    enum's raw value is inferred the same way as the other type-name guesses
    in this file, not independently confirmed.
    """
    for entry in obs_dict.get("logs") or []:
        if (isinstance(entry, dict) and entry.get("type") == "Attack"
                and entry.get("playerIndex") == player_index):
            return entry.get("attackId")
    return None


def _items_played_count(prev_obs, cur_obs, player_index):
    return sum(1 for cid in _newly_discarded_ids(prev_obs, cur_obs, player_index) if _is_item_card(cid))


def _boss_setup_reward(prev_obs, cur_obs, me_index, opp_index):
    """Bonus for a Boss's Orders (detected via _newly_discarded_ids, not a
    guessed log event) that pulls up either a 2+ prize target Jetting Blow
    (120 dmg) can now finish, or a tankier target within Nebula Beam / Mega
    Froslass ex attack range (<=210). This is the roughest approximation
    added in this file: it reads the post-effect board only (no true credit
    assignment to whichever later attack actually lands the KO, which may
    happen on a different decision step), ignores weakness/resistance on the
    120 threshold, and _prize_value_by_id needs CARD_DB (defaults every
    unknown target to 1 prize if that's unavailable).
    """
    if BOSS_ID not in _newly_discarded_ids(prev_obs, cur_obs, me_index):
        return 0.0
    target = _active_pokemon(cur_obs, opp_index)
    if not target:
        return 0.0
    remaining_hp = target.get("hp")
    if not isinstance(remaining_hp, (int, float)):
        return 0.0
    if remaining_hp <= 120 and _prize_value_by_id(target.get("id")) >= 2:
        return BOSS_MULTI_PRIZE_REWARD
    if 120 < remaining_hp <= 210:
        return BOSS_BIG_FINISH_REWARD
    return 0.0


_turns_taken = {0: 0, 1: 0}


def reset_turn_tracking():
    """Call at the start of every new game (both self-play and
    heuristic-opponent mode) -- see _no_attack_turn_penalty."""
    global _turns_taken
    _turns_taken = {0: 0, 1: 0}


def _no_attack_turn_penalty(prev_obs, cur_obs, me_index):
    """Penalize ending our own turn (turn control passing to the opponent)
    without attacking, from our 2nd turn onward. Turn-passing is read off
    current.yourIndex flipping between prev_obs and cur_obs -- the same field
    this module already keys "me" off of, so more solidly grounded than a
    guessed log-event name, but it still assumes every yourIndex flip is a
    full turn change (not, say, a mid-turn prompt directed at the opponent)
    and that an Attack log entry always appears in the flipping step's delta
    when we do attack (matches the "attack ends the turn" rule noted in
    starmie_agent.py's score-priority comment).

    Needs per-player state across steps -- this can't be answered from one
    (prev_obs, cur_obs) pair alone. Only correct for one game running in this
    process at a time; reset_turn_tracking() must be called on every new
    game, and this will misbehave if multiple games run concurrently through
    a shared process without keying the state by game/env id.
    """
    cur_mover = (cur_obs.get("current") or {}).get("yourIndex")
    if cur_mover == me_index:
        return 0.0
    _turns_taken[me_index] = _turns_taken.get(me_index, 0) + 1
    attacked = any(
        isinstance(e, dict) and e.get("type") == "Attack" and e.get("playerIndex") == me_index
        for e in (cur_obs.get("logs") or [])
    )
    if _turns_taken[me_index] >= 2 and not attacked:
        return NO_ATTACK_PENALTY
    return 0.0


# ── Follow-up shaping-term helpers ──

def _hand_size(obs_dict, player_index):
    players = _players(obs_dict)
    if player_index >= len(players):
        return 0
    return len((players[player_index] or {}).get("hand") or [])


def _deck_remaining(obs_dict, player_index):
    """Same list-length-as-count convention as _prizes_remaining (deck
    contents are hidden too, but the count is public in real TCG rules)."""
    players = _players(obs_dict)
    if player_index >= len(players):
        return 0
    return len((players[player_index] or {}).get("deck") or [])


def _mega_starmie_available(obs_dict, player_index):
    """Mega Starmie ex in play (active or bench) with at least 1 Energy
    attached -- able to Jetting Blow this turn. Mirrors starmie_agent.py's
    bench_ready_mega check."""
    return any(
        m.get("id") == STARMIE_ID and len(m.get("energyCards") or []) >= 1
        for m in _pokemon_in_play(obs_dict, player_index)
    )


def _is_supporter_card(card_id):
    if not CARD_DB or CardType is None or card_id is None:
        return None
    data = CARD_DB.get(card_id)
    ct = getattr(data, "cardType", None) if data else None
    return None if ct is None else ct == getattr(CardType, "SUPPORTER", None)


def _supporter_played_count(prev_obs, cur_obs, player_index):
    """Supporters played this step, Wally's Compassion included -- see
    WALLY_HEAL_BASE for why that's fine (its magnitude already keeps a
    low-value heal net-negative with this bonus stacked on top)."""
    return sum(
        1 for cid in _newly_discarded_ids(prev_obs, cur_obs, player_index)
        if _is_supporter_card(cid)
    )


def _stadium_identity(obs_dict):
    """(cardId, ownerPlayerIndex) of the Stadium currently in play, or None.
    Reads obs["current"]["stadium"] + each entry's playerIndex, the same
    zone/field starmie_agent.py's deck_counts() already reads -- though that
    function accesses playerIndex via a defensive getattr(..., fallback), so
    treat its presence here as likely but not fully confirmed."""
    stadium = (obs_dict.get("current") or {}).get("stadium") or []
    s = stadium[0] if stadium else None
    return (s.get("id"), s.get("playerIndex")) if s else None


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
        energy-attach, damage, draw, Wally-heal, Ignition-waste, Crushing
        Hammer, Budew/Item, Boss's Orders, bench-size, matchup-evolution,
        no-attack-turn, Resentful-Refrain-timing, Absolute-Snow-into-tank,
        deck-out-risk/Run-Away-Draw, Supporter-play, and Stadium terms below.
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

    draw_reward = DRAW_REWARD_PER_CARD * _draw_count(cur_obs, me_index)

    healed = _heal_dealt(cur_obs, me_index)
    wally_reward = (WALLY_HEAL_BASE + WALLY_HEAL_PER_10HP * (healed / 10)) if healed > 0 else 0.0

    used_attack = _attack_id_used(cur_obs, me_index)

    ignition_penalty = 0.0
    if used_attack is not None and _active_has_ignition(prev_obs, me_index):
        if NEBULA_BEAM_ID is not None and used_attack != NEBULA_BEAM_ID:
            ignition_penalty = IGNITION_WASTE_PENALTY

    froslass_attack_penalty = 0.0
    froslass_attack_bonus = 0.0
    if used_attack is not None and used_attack in (RESENTFUL_REFRAIN_ID, ABSOLUTE_SNOW_ID):
        refrain_damage = 50 * _hand_size(prev_obs, opp_index)
        if used_attack == RESENTFUL_REFRAIN_ID:
            if refrain_damage <= 100 and _mega_starmie_available(prev_obs, me_index):
                froslass_attack_penalty = RESENTFUL_REFRAIN_LOW_VALUE_PENALTY
        elif used_attack == ABSOLUTE_SNOW_ID:
            snow_target = _active_pokemon(prev_obs, opp_index)
            target_hp = snow_target.get("hp") if snow_target else None
            if isinstance(target_hp, (int, float)) and target_hp > refrain_damage:
                froslass_attack_bonus = ABSOLUTE_SNOW_HIGH_HP_REWARD

    deck_save_reward = 0.0
    if _deck_remaining(prev_obs, me_index) <= DECK_OUT_RISK_THRESHOLD:
        ran_away = (_count_in_play(prev_obs, me_index, DUDUNSPARCE_ID)
                    > _count_in_play(cur_obs, me_index, DUDUNSPARCE_ID))
        if ran_away:
            deck_save_reward = RUN_AWAY_DRAW_DECK_SAVE_REWARD

    supporter_reward = SUPPORTER_PLAY_REWARD * _supporter_played_count(prev_obs, cur_obs, me_index)

    stadium_reward = 0.0
    prev_stadium = _stadium_identity(prev_obs)
    cur_stadium = _stadium_identity(cur_obs)
    if cur_stadium is not None and cur_stadium[1] == me_index and cur_stadium != prev_stadium:
        stadium_reward = STADIUM_PLAY_REWARD
        if prev_stadium is not None and prev_stadium[1] == opp_index:
            stadium_reward += STADIUM_BUMP_REWARD

    hammer_played = HAMMER_ID in _newly_discarded_ids(prev_obs, cur_obs, me_index)
    opp_energy_lost = any(_is_energy_card(cid) for cid in _newly_discarded_ids(prev_obs, cur_obs, opp_index))
    hammer_reward = HAMMER_DISCARD_REWARD if (hammer_played and opp_energy_lost) else 0.0

    budew_reward = 0.0
    if BUDEW_IDS and _opponent_has_any(prev_obs, BUDEW_IDS, opp_index):
        budew_reward = ITEM_VS_BUDEW_REWARD * _items_played_count(prev_obs, cur_obs, me_index)

    boss_reward = _boss_setup_reward(prev_obs, cur_obs, me_index, opp_index)

    basics_in_play = _count_in_play(cur_obs, me_index, STARYU_ID) + _count_in_play(cur_obs, me_index, SNORUNT_ID)
    if basics_in_play > 5:
        bench_penalty = BENCH_OVER_5_PENALTY
    elif basics_in_play > 4:
        bench_penalty = BENCH_OVER_4_PENALTY
    else:
        bench_penalty = 0.0

    evolve_matchup_penalty = 0.0
    starmie_evolved = _count_in_play(cur_obs, me_index, STARMIE_ID) > _count_in_play(prev_obs, me_index, STARMIE_ID)
    froslass_evolved = _count_in_play(cur_obs, me_index, FROSLASS_ID) > _count_in_play(prev_obs, me_index, FROSLASS_ID)
    if starmie_evolved and IONO_BELLIBOLT_IDS and _opponent_has_any(prev_obs, IONO_BELLIBOLT_IDS, opp_index):
        evolve_matchup_penalty += IONO_BELLIBOLT_EVOLVE_PENALTY
    if (froslass_evolved and ARCHALUDON_IDS and _opponent_has_any(prev_obs, ARCHALUDON_IDS, opp_index)
            and _prizes_remaining(prev_obs, me_index) > 2):
        evolve_matchup_penalty += ARCHALUDON_EVOLVE_PENALTY

    no_attack_penalty = _no_attack_turn_penalty(prev_obs, cur_obs, me_index)

    return (
        my_prize_reward
        - opp_prize_reward
        + evolve_reward
        + energy_reward
        + damage_reward
        + draw_reward
        + wally_reward
        - ignition_penalty
        + hammer_reward
        + budew_reward
        + boss_reward
        - bench_penalty
        - evolve_matchup_penalty
        - no_attack_penalty
        - froslass_attack_penalty
        + froslass_attack_bonus
        + deck_save_reward
        + supporter_reward
        + stadium_reward
    )
