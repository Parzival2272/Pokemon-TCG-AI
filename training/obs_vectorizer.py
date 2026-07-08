import numpy as np

MAX_BENCH = 5
MAX_HAND = 60
# Some MAIN-select decisions (PLAY/ATTACH/EVOLVE/ABILITY/DISCARD/RETREAT/
# ATTACK/END all share one option list) can offer more options than a
# simple per-card cap would suggest, so this needs real headroom.
MAX_OPTIONS = 128
CARD_FEATURES = 4

VECTOR_SIZE = (
    CARD_FEATURES * 2  # your + opponent active
    + CARD_FEATURES * MAX_BENCH * 2  # your + opponent bench
    + MAX_HAND  # hand card ids
    + 4  # your_prizes, opp_prizes, stadium_id, context
    + MAX_OPTIONS  # option_types
)


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
        + [your_prizes, opp_prizes, stadium_id, context]  # 4
        + option_types  # MAX_OPTIONS
    )
    # Total: VECTOR_SIZE features
    return np.array(vec, dtype=np.float32)
