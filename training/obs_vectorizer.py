import numpy as np
from collections import Counter

MAX_BENCH = 5
MAX_HAND = 60
MAX_DECK_SLOTS = 60
# Some MAIN-select decisions (PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/RETREAT/
# ATTACK/END all share one option list) can offer more options than a
# simple per-card cap would suggest, so this needs real headroom.
MAX_OPTIONS = 128
CARD_FEATURES = 4

VECTOR_SIZE = (
    CARD_FEATURES * 2  # your + opponent active
    + CARD_FEATURES * MAX_BENCH * 2  # your + opponent bench
    + MAX_HAND  # hand card ids
    + MAX_DECK_SLOTS  # per-deck-slot deck visibility flags
    + 4  # your_prizes, opp_prizes, stadium_id, context
    + MAX_OPTIONS  # option_types
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


def obs_to_vector(obs_dict: dict) -> np.ndarray:
    current = obs_dict.get("current") or {}
    your_index = current.get("yourIndex", 0)
    players = current.get("players", [{}, {}])
    you = players[your_index] if len(players) > your_index else {}
    opp = players[1 - your_index] if len(players) > 1 else {}

    def pokemon_vec(mon):
        if not mon:
            return [0, 0, 0, 0]
        return [
            mon.get("id", 0) / 1300,  # normalized card id
            mon.get("hp", 0) / 480,  # normalized hp
            mon.get("maxHp", 1) / 480,  # normalized max hp
            len(mon.get("energies", [])) / 20,  # normalized energy count
        ]

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
    hand_vec = [c.get("id", 0) / 2000 if c else 0 for c in hand]
    hand_vec += [0] * (MAX_HAND - len(hand_vec))

    # Prize flag by deck slot (60 features in {0, 1}).
    deck_slot_flags = prize_flag_vec(obs_dict)

    # Prize counts
    your_prizes = len(you.get("prize") or []) / 6
    opp_prizes = len(opp.get("prize") or []) / 6

    # Stadium
    stadium = current.get("stadium") or []
    stadium_id = (stadium[0].get("id", 0) if stadium else 0) / 2000

    # Select context + option types (what kind of decision is this?)
    select = obs_dict.get("select") or {}
    context = (select.get("context") or 0) / 50
    options = select.get("option") or []
    option_types = [o.get("type", 0) / 16 for o in options[:MAX_OPTIONS]]
    option_types += [0] * (MAX_OPTIONS - len(option_types))

    vec = (
        pokemon_vec(your_active)  # 4
        + pokemon_vec(opp_active)  # 4
        + [f for m in your_bench for f in pokemon_vec(m)]  # 20
        + [f for m in opp_bench for f in pokemon_vec(m)]  # 20
        + hand_vec  # MAX_HAND
        + deck_slot_flags  # MAX_DECK_SLOTS
        + [your_prizes, opp_prizes, stadium_id, context]  # 4
        + option_types  # MAX_OPTIONS
    )
    # Total: VECTOR_SIZE features
    return np.array(vec, dtype=np.float32)
