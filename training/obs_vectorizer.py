import numpy as np
from collections import Counter

MAX_BENCH = 5
MAX_HAND = 60
MAX_DECK_SLOTS = 60
# Some MAIN-select decisions (PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/RETREAT/
# ATTACK/END all share one option list) can offer more options than a
# simple per-card cap would suggest, so this needs real headroom.
MAX_OPTIONS = 128

# All card-id features (Pokemon ids, hand ids, attack ids, ...) share one
# normaliser so the network sees a consistent scale. Max observed id is well
# under this (trainers ~1260), keeping features in [0, 1].
CARD_ID_NORM = 2000.0

# Per-Pokemon features: id, hp, maxHp, energy count, appeared-this-turn,
# has-tool. (Fields sourced from the engine's PokemonJson in ToJson.h.)
POKE_FEATURES = 6
# Energy-type histogram length (EnergyType enum: COLORLESS..TEAM_ROCKET = 0..11).
N_ENERGY_TYPES = 12
# Per-option features: option type, the card the option acts on, attack id.
OPTION_FEATURES = 3
# Special conditions per player's active: poisoned/burned/asleep/paralyzed/confused.
COND_FEATURES = 5
# Turn / resource / count scalars (see _state_block).
STATE_FEATURES = 16
# Log-derived temporal context (see _log_block): damage dealt/taken and each
# side's last attack id, summarising events since the previous observation.
LOG_FEATURES = 4

VECTOR_SIZE = (
    POKE_FEATURES * 2  # your + opponent active
    + POKE_FEATURES * MAX_BENCH * 2  # your + opponent bench
    + N_ENERGY_TYPES * 2  # your + opponent active energy-type histograms
    + MAX_HAND  # hand card ids
    + MAX_DECK_SLOTS  # per-deck-slot deck visibility flags
    + COND_FEATURES * 2  # your + opponent special conditions
    + STATE_FEATURES  # turn / resource / selection-count scalars
    + LOG_FEATURES  # recent damage / last attacks from the event log
    + 4  # your_prizes, opp_prizes, stadium_id, context
    + OPTION_FEATURES * MAX_OPTIONS  # per-option [type, card id, attack id]
)

# Deck order
VECTORIZER_DECK = [0] * MAX_DECK_SLOTS


def set_vectorizer_deck(deck: list[int]) -> None:
    """Set the 60 card deck order used by the deck slot flags."""
    global VECTORIZER_DECK
    VECTORIZER_DECK = list(deck)


def visible_owned_card_ids(current: dict, your_index: int) -> list[int]:
    """Collect visible card IDs known to be on your side and not in prizes."""
    players = current.get("players", [{}, {}])
    you = players[your_index] if len(players) > your_index else {}
    card_ids: list[int] = []

    for c in (you.get("hand") or []):
        cid = c.get("id") if c else None
        if cid:
            card_ids.append(cid)

    for c in (you.get("discard") or []):
        cid = c.get("id") if c else None
        if cid:
            card_ids.append(cid)

    for mon in (you.get("active") or []):
        if not mon:
            continue
        mid = mon.get("id")
        if mid:
            card_ids.append(mid)
        for ec in (mon.get("energyCards") or []):
            cid = ec.get("id") if ec else None
            if cid:
                card_ids.append(cid)
        for tc in (mon.get("tools") or []):
            cid = tc.get("id") if tc else None
            if cid:
                card_ids.append(cid)
        for pe in (mon.get("preEvolution") or []):
            cid = pe.get("id") if pe else None
            if cid:
                card_ids.append(cid)

    for mon in (you.get("bench") or []):
        if not mon:
            continue
        mid = mon.get("id")
        if mid:
            card_ids.append(mid)
        for ec in (mon.get("energyCards") or []):
            cid = ec.get("id") if ec else None
            if cid:
                card_ids.append(cid)
        for tc in (mon.get("tools") or []):
            cid = tc.get("id") if tc else None
            if cid:
                card_ids.append(cid)
        for pe in (mon.get("preEvolution") or []):
            cid = pe.get("id") if pe else None
            if cid:
                card_ids.append(cid)

    for c in (current.get("stadium") or []):
        if not c or c.get("playerIndex") != your_index:
            continue
        cid = c.get("id")
        if cid:
            card_ids.append(cid)

    for c in (current.get("looking") or []):
        if not c or c.get("playerIndex") != your_index:
            continue
        cid = c.get("id")
        if cid:
            card_ids.append(cid)

    return card_ids


def revealed_deck_card_ids(obs_dict: dict, your_index: int) -> list[int]:
    """Collect cards currently revealed from your deck."""
    select = obs_dict.get("select") or {}
    revealed: list[int] = []
    select_deck = select.get("deck") or []

    for c in select_deck:
        cid = c.get("id") if c else None
        if cid:
            revealed.append(cid)

    for o in (select.get("option") or []):
        if o.get("playerIndex") != your_index:
            continue
        if o.get("area") != 1:  # AreaType.DECK
            continue
        cid = o.get("cardId")
        if cid:
            revealed.append(cid)

    return revealed


def prize_flag_vec(obs_dict: dict) -> list[float]:
    """Return 60 boolean deck slot flags.
    """
    current = obs_dict.get("current") or {}
    your_index = int(current.get("yourIndex", 0) or 0)

    if not VECTORIZER_DECK:
        return [0.0] * MAX_DECK_SLOTS

    visible_non_prize = Counter(visible_owned_card_ids(current, your_index))
    visible_non_prize.update(revealed_deck_card_ids(obs_dict, your_index))

    slot_flags: list[float] = []
    non_prize_budget = Counter(visible_non_prize)
    for cid in VECTORIZER_DECK:
        if cid and non_prize_budget[cid] > 0:
            slot_flags.append(0.0)
            non_prize_budget[cid] -= 1
        else:
            slot_flags.append(1.0)

    return slot_flags


# --- Per-option card resolution ---------------------------------------------
# The engine (ApiJson.h SelectOptionJson) encodes each option's card by
# reference, not by id, so to feed the network "which card does this option
# act on" we resolve it the same way the heuristics' get_card does. Option
# type ints match SelectOptionType in ApiType.h.
_OPT_PLAY = 7  # card is hand[index]
_OPT_ATTACK = 13  # no card; carries attackId
_OPT_SKILL = 15  # carries cardId directly
# Types that carry an (area, index) reference (Card/ToolCard/EnergyCard/Energy
# also carry playerIndex; Attach/Evolve/Ability/Discard are the acting player).
_OPT_AREA_INDEX = {3, 4, 5, 6, 8, 9, 10, 11}


def _zone_list(obs_dict: dict, current: dict, area, player_index: int) -> list:
    """The card list a given AreaType int refers to, for `player_index`."""
    players = current.get("players") or [{}, {}]
    ps = players[player_index] if 0 <= player_index < len(players) else {}
    if area == 2:  # HAND
        return ps.get("hand") or []
    if area == 3:  # DISCARD
        return ps.get("discard") or []
    if area == 4:  # ACTIVE
        return ps.get("active") or []
    if area == 5:  # BENCH
        return ps.get("bench") or []
    if area == 6:  # PRIZE (face-down entries are null)
        return ps.get("prize") or []
    if area == 1:  # DECK (only populated during a deck select)
        return (obs_dict.get("select") or {}).get("deck") or []
    if area == 7:  # STADIUM
        return current.get("stadium") or []
    if area == 12:  # LOOKING
        return current.get("looking") or []
    return []


def _option_card_id(obs_dict: dict, current: dict, opt: dict, your_index: int) -> int:
    """Resolve the card id an option acts on (0 if it references no card)."""
    t = opt.get("type")
    if t == _OPT_SKILL:
        return opt.get("cardId") or 0
    if t == _OPT_PLAY:
        lst = _zone_list(obs_dict, current, 2, your_index)  # HAND
        idx = opt.get("index")
    elif t in _OPT_AREA_INDEX:
        pi = opt.get("playerIndex")
        pi = your_index if pi is None else pi
        lst = _zone_list(obs_dict, current, opt.get("area"), pi)
        idx = opt.get("index")
    else:
        return 0
    if idx is None or not (0 <= idx < len(lst)):
        return 0
    card = lst[idx]
    return (card.get("id") or 0) if card else 0


def _poke_vec(mon) -> list[float]:
    if not mon:
        return [0.0] * POKE_FEATURES
    return [
        (mon.get("id") or 0) / CARD_ID_NORM,
        (mon.get("hp") or 0) / 480,
        (mon.get("maxHp") or 1) / 480,
        len(mon.get("energies") or []) / 20,
        1.0 if mon.get("appearThisTurn") else 0.0,
        1.0 if (mon.get("tools") or []) else 0.0,
    ]


def _energy_hist(mon) -> list[float]:
    """Count of each EnergyType attached to `mon` (normalized)."""
    hist = [0.0] * N_ENERGY_TYPES
    if not mon:
        return hist
    for e in (mon.get("energies") or []):
        if isinstance(e, int) and 0 <= e < N_ENERGY_TYPES:
            hist[e] += 1.0
    return [h / 8.0 for h in hist]


def _cond_vec(player: dict) -> list[float]:
    """Special conditions on a player's Active Pokemon (5 flags)."""
    return [
        1.0 if player.get("poisoned") else 0.0,
        1.0 if player.get("burned") else 0.0,
        1.0 if player.get("asleep") else 0.0,
        1.0 if player.get("paralyzed") else 0.0,
        1.0 if player.get("confused") else 0.0,
    ]


# Log event-type ints (LogType enum in ApiType.h): Attack=15, HpChange=16.
_LOG_ATTACK = 15
_LOG_HPCHANGE = 16


def _log_block(logs, your_index: int) -> list[float]:
    """Temporal context from events since the previous observation:
    damage dealt to the opponent, damage taken, and each side's last attack id.
    HpChange `value` is the HP delta (negative == damage); `playerIndex` is
    whose Pokemon changed. Gives the otherwise-stateless observation short-term
    memory (e.g. "the opponent just hit my Active for 240")."""
    opp_index = 1 - your_index
    dmg_dealt = 0.0
    dmg_taken = 0.0
    my_attack = 0
    opp_attack = 0
    for log in (logs or []):
        t = log.get("type")
        if t == _LOG_HPCHANGE:
            v = log.get("value") or 0
            if v < 0:
                pi = log.get("playerIndex")
                if pi == opp_index:
                    dmg_dealt += -v
                elif pi == your_index:
                    dmg_taken += -v
        elif t == _LOG_ATTACK:
            pi = log.get("playerIndex")
            if pi == your_index:
                my_attack = log.get("attackId") or 0
            elif pi == opp_index:
                opp_attack = log.get("attackId") or 0
    return [
        min(dmg_dealt / 480, 1.0),
        min(dmg_taken / 480, 1.0),
        my_attack / CARD_ID_NORM,
        opp_attack / CARD_ID_NORM,
    ]


def _state_block(current: dict, select: dict, you: dict, opp: dict, your_index: int) -> list[float]:
    """Turn/resource/selection-count scalars (STATE_FEATURES long)."""
    return [
        (current.get("turn") or 0) / 100,
        (current.get("turnActionCount") or 0) / 50,
        1.0 if current.get("firstPlayer") == your_index else 0.0,
        1.0 if current.get("supporterPlayed") else 0.0,
        1.0 if current.get("stadiumPlayed") else 0.0,
        1.0 if current.get("energyAttached") else 0.0,
        1.0 if current.get("retreated") else 0.0,
        (you.get("deckCount") or 0) / 60,
        (opp.get("deckCount") or 0) / 60,
        (opp.get("handCount") or 0) / 60,
        (you.get("benchMax") or 0) / 8,
        (opp.get("benchMax") or 0) / 8,
        (select.get("minCount") or 0) / 16,
        (select.get("maxCount") or 0) / 16,
        (select.get("remainDamageCounter") or 0) / 20,
        (select.get("remainEnergyCost") or 0) / 8,
    ]


def obs_to_vector(obs_dict: dict) -> np.ndarray:
    current = obs_dict.get("current") or {}
    your_index = current.get("yourIndex", 0)
    players = current.get("players", [{}, {}])
    you = players[your_index] if len(players) > your_index else {}
    opp = players[1 - your_index] if len(players) > 1 else {}
    select = obs_dict.get("select") or {}

    # Active pokemon (yours + opponent's)
    your_active = (you.get("active") or [None])[0]
    opp_active = (opp.get("active") or [None])[0]

    # Bench (padded to MAX_BENCH)
    your_bench = (you.get("bench") or [])[:MAX_BENCH]
    your_bench += [None] * (MAX_BENCH - len(your_bench))
    opp_bench = (opp.get("bench") or [])[:MAX_BENCH]
    opp_bench += [None] * (MAX_BENCH - len(opp_bench))

    # Hand (padded to MAX_HAND, just card ids)
    hand = (you.get("hand") or [])[:MAX_HAND]
    hand_vec = [(c.get("id", 0) / CARD_ID_NORM) if c else 0.0 for c in hand]
    hand_vec += [0.0] * (MAX_HAND - len(hand_vec))

    # Deck-slot visibility flags (60 features in {0, 1}).
    deck_slot_flags = prize_flag_vec(obs_dict)

    # Prize counts
    your_prizes = len(you.get("prize") or []) / 6
    opp_prizes = len(opp.get("prize") or []) / 6

    # Stadium
    stadium = current.get("stadium") or []
    stadium_id = (stadium[0].get("id", 0) if stadium else 0) / CARD_ID_NORM

    # Select context (what kind of decision is this?)
    context = (select.get("context") or 0) / 50

    # Per-option block: type, the card the option acts on, and attack id.
    options = select.get("option") or []
    opt_block: list[float] = []
    for o in options[:MAX_OPTIONS]:
        opt_block.append((o.get("type") or 0) / 16)
        opt_block.append(_option_card_id(obs_dict, current, o, your_index) / CARD_ID_NORM)
        opt_block.append((o.get("attackId") or 0) / CARD_ID_NORM)
    opt_block += [0.0] * (OPTION_FEATURES * MAX_OPTIONS - len(opt_block))

    vec = (
        _poke_vec(your_active)  # POKE_FEATURES
        + _poke_vec(opp_active)  # POKE_FEATURES
        + [f for m in your_bench for f in _poke_vec(m)]  # POKE_FEATURES * MAX_BENCH
        + [f for m in opp_bench for f in _poke_vec(m)]  # POKE_FEATURES * MAX_BENCH
        + _energy_hist(your_active)  # N_ENERGY_TYPES
        + _energy_hist(opp_active)  # N_ENERGY_TYPES
        + hand_vec  # MAX_HAND
        + deck_slot_flags  # MAX_DECK_SLOTS
        + _cond_vec(you)  # COND_FEATURES
        + _cond_vec(opp)  # COND_FEATURES
        + _state_block(current, select, you, opp, your_index)  # STATE_FEATURES
        + _log_block(obs_dict.get("logs"), your_index)  # LOG_FEATURES
        + [your_prizes, opp_prizes, stadium_id, context]  # 4
        + opt_block  # OPTION_FEATURES * MAX_OPTIONS
    )
    assert len(vec) == VECTOR_SIZE, (len(vec), VECTOR_SIZE)
    return np.array(vec, dtype=np.float32)
