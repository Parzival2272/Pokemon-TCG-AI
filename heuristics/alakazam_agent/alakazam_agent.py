import os
import sys
from collections import defaultdict
from ptcg.api import (
    AreaType, CardType, EnergyType, Observation,
    SelectContext, OptionType, Card, Pokemon,
    all_card_data, to_observation_class
)

# Load deck.csv: package-local first, then the Kaggle agent directory.
_deck_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
if not os.path.exists(_deck_path):
    _deck_path = "/kaggle_simulations/agent/deck.csv"
with open(_deck_path) as _f:
    my_deck: list[int] = [int(_line) for _line in _f.read().splitlines() if _line.strip()]


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py."""
    global my_deck
    my_deck = list(deck_list)

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}

# --- Your Decklist IDs ---
Abra = 741
Kadabra = 742
Alakazam = 743
Dunsparce = 305
Dudunsparce = 66
Fezandipiti_ex = 140
Genesect = 142
Shaymin = 343

Rare_Candy = 1079
Enhanced_Hammer = 1081
Buddy_Buddy_Poffin = 1086
Night_Stretcher = 1097
Sacred_Ash = 1129
Poke_Pad = 1152
Lucky_Helmet = 1156
Boss_Orders = 1182
Brocks_Scouting = 1210
Hilda = 1225
Dawn = 1231
Lanas_Aid = 1184

Basic_Psychic_Energy = 5
Telepath_Psychic_Energy = 19
Enriching_Energy = 13

# Opponent card IDs to watch for
Duskull = 131
Slowpoke_IDs = (162, 327)
Froakie_IDs = (33, 945)
Wellspring_Mask_Ogerpon_ex = 108
N_Darumaka = 257
Mist_Energy = 11
Rock_Fighting_Energy = 20

# Attack IDs
ATTACK_TELEPORTATION = 1070
ATTACK_SUPER_PSY_BOLT = 1071
ATTACK_POWERFUL_HAND = 1072

# Sets for easy filtering
ABRA_LINE = {Abra, Kadabra, Alakazam}
DUNSPARCE_LINE = {Dunsparce, Dudunsparce}
PSYCHIC_ENERGY_IDS = {Basic_Psychic_Energy, Telepath_Psychic_Energy}
RECOVERY_CARDS = {Night_Stretcher, Sacred_Ash, Brocks_Scouting, Lanas_Aid}

pre_turn = 0
ability_used_dudunsparce = False
ability_used_fezandipiti = False

def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK: return obs.select.deck[index]
        case AreaType.HAND: return ps.hand[index]
        case AreaType.DISCARD: return ps.discard[index]
        case AreaType.ACTIVE: return ps.active[index]
        case AreaType.BENCH: return ps.bench[index]
        case AreaType.PRIZE: return ps.prize[index]
        case AreaType.STADIUM: return obs.current.stadium[index]
        case AreaType.LOOKING: return obs.current.looking[index]
        case _: return None

def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12: count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name: count -= 1
    return max(0, count)

def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None: return my_deck

    state = obs.current
    select = obs.select
    context = select.context

    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize_count = len(my_state.prize)

    global pre_turn, ability_used_dudunsparce, ability_used_fezandipiti
    if pre_turn != state.turn:
        pre_turn = state.turn
        ability_used_dudunsparce = False
        ability_used_fezandipiti = False

    # Count cards across zones
    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    my_field = []

    for card in my_state.active:
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((0, card))
    for idx, card in enumerate(my_state.bench):
        if card is not None:
            field_counts[card.id] += 1
            my_field.append((idx + 1, card))

    for card in my_state.hand:
        hand_counts[card.id] += 1

    abra_line_on_field = field_counts[Abra] + field_counts[Kadabra] + field_counts[Alakazam]
    dunsparce_line_on_field = field_counts[Dunsparce] + field_counts[Dudunsparce]

    # Opponent analysis
    op_all_pokemon = []
    for card in op_state.active + op_state.bench:
        if card is not None: op_all_pokemon.append(card)

    op_has_duskull = any(p.id == Duskull for p in op_all_pokemon)
    op_has_water_threat = any(
        p.id in Slowpoke_IDs or p.id in Froakie_IDs or p.id == Wellspring_Mask_Ogerpon_ex or p.id == N_Darumaka
        for p in op_all_pokemon
    )

    op_used_ace_spec = any(
        card_table.get(log.cardId).aceSpec for log in obs.logs
        if hasattr(log, 'cardId') and log.cardId is not None and card_table.get(log.cardId)
        and hasattr(log, 'playerIndex') and log.playerIndex == (1 - my_index)
    )

    bench_free = my_state.benchMax - len(my_state.bench)
    active_pokemon = my_state.active[0] if my_state.active else None
    active_id = active_pokemon.id if active_pokemon else 0
    op_active_hp = op_state.active[0].hp if op_state.active and op_state.active[0] else 999
    active_has_psychic = any(ec.id in PSYCHIC_ENERGY_IDS for ec in active_pokemon.energyCards) if active_pokemon else False

    # Mathematical Lethal Assessment
    use_kadabra_finish = (op_active_hp <= 30)
    current_hand_size = len(my_state.hand)
    current_damage = current_hand_size * 20

    max_potential_hand = current_hand_size
    if not ability_used_dudunsparce and field_counts[Dudunsparce] > 0: max_potential_hand += 3
    if not ability_used_fezandipiti and field_counts[Fezandipiti_ex] > 0: max_potential_hand += 3
    if not state.supporterPlayed:
        if hand_counts[Dawn] > 0: max_potential_hand += 2
        elif hand_counts[Hilda] > 0: max_potential_hand += 1

    max_potential_damage = max_potential_hand * 20

    target_use_boss = False
    target_can_kill = False
    target_pokemon = None
    target_idx = -1
    target_prize_gain = 0
    need_dudunsparce_draw = False
    need_fezandipiti_draw = False
    need_fezandipiti_for_setup = False
    need_retreat_energy = False

    if op_active_hp <= max_potential_damage:
        target_pokemon = op_state.active[0]
        target_can_kill = True
        target_prize_gain = prize_count(target_pokemon)
        if op_active_hp > current_damage:
            need_dudunsparce_draw = True
            need_fezandipiti_draw = True
    else:
        best_bench_prize = -1
        best_bench_idx = -1
        for idx, p in enumerate(op_state.bench):
            if p is None: continue
            if p.hp <= (max_potential_hand - 1) * 20: # -1 hand size for playing boss
                p_prize = prize_count(p)
                if p_prize > best_bench_prize:
                    best_bench_prize = p_prize
                    best_bench_idx = idx + 1

        if best_bench_prize > 0 and hand_counts[Boss_Orders] > 0:
            target_use_boss = True
            target_can_kill = True
            target_idx = best_bench_idx
            target_pokemon = op_state.bench[best_bench_idx - 1]
            target_prize_gain = best_bench_prize
            if target_pokemon.hp > (current_hand_size - 1) * 20:
                need_dudunsparce_draw = True
                need_fezandipiti_draw = True
        elif best_bench_prize > 0 and hand_counts[Boss_Orders] == 0:
            need_fezandipiti_for_setup = True

    can_win_this_turn = target_can_kill and my_prize_count <= target_prize_gain
    safe_draws = my_state.deckCount - my_prize_count - 1 if not can_win_this_turn else 999

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER: score = o.number
        elif o.type == OptionType.YES: score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is None:
                scores.append(score)
                continue

            energy_count = len(card.energies) if isinstance(card, Pokemon) else 0

            if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                if o.playerIndex == my_index:
                    if card.id == Alakazam: score += 100 + energy_count * 10
                    elif card.id == Kadabra: score += 90 if (op_active_hp <= 30) else 30
                    elif card.id == Abra: score += 10
                    elif card.id in DUNSPARCE_LINE: score += 5
                    else: score += 1
                else:
                    if target_use_boss and target_pokemon is not None:
                        if o.index == target_idx - 1: score += 100

            elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                if card.id == Abra: score = 10
                elif card.id == Dunsparce: score = 5
                elif card.id == Shaymin: score = 1

            elif context == SelectContext.SETUP_BENCH_POKEMON:
                if card.id == Abra:
                    score = 200 if abra_line_on_field == 0 else 100 + (3 - abra_line_on_field) * 10
                elif card.id == Dunsparce:
                    score = 150 if dunsparce_line_on_field == 0 else 50

            elif context == SelectContext.TO_HAND:
                score = 200 - hand_counts.get(card.id, 0) * 50
                if card.id == Dudunsparce: score += 80 if (field_counts[Dunsparce] >= 1 and field_counts[Dudunsparce] == 0) else -50
                elif card.id == Kadabra: score += 70 if field_counts[Abra] >= 1 else -20
                elif card.id == Alakazam: score += 60 if (field_counts[Kadabra] >= 1 or field_counts[Abra] >= 1) else -20
                elif card.id == Abra: score += 50 if abra_line_on_field < 3 else -50
                elif card.id in PSYCHIC_ENERGY_IDS: score += 30 if not state.energyAttached else -10
                elif card.id == Rare_Candy: score += 40 if field_counts[Abra] >= 1 else -10

            elif context == SelectContext.ATTACH_FROM:
                if isinstance(card, Pokemon):
                    if len(card.energyCards) >= 1: score = -1
                    elif card.id in ABRA_LINE:
                        score = 100
                        if card.id == Alakazam: score += 20
                        if o.area == AreaType.ACTIVE: score += 5
                    elif card.id in DUNSPARCE_LINE: score = 50

            elif context == SelectContext.TO_BENCH:
                if card.id == Abra: score = 100
                elif card.id == Dunsparce: score = 80
                elif card.id == Shaymin: score = 40 if op_has_water_threat else -1

        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            is_early = state.turn <= 2

            if card.id == Buddy_Buddy_Poffin:
                if abra_line_on_field < 3 or dunsparce_line_on_field < 2: score = 20000
                else: score = -1

            elif card.id == Abra:
                if is_early: score += 500
                elif abra_line_on_field < 3: score += 200
                elif bench_free <= 1: score = -1
                else: score += 50

            elif card.id == Dunsparce:
                if dunsparce_line_on_field < 1: score += 400 if is_early else 100
                elif dunsparce_line_on_field < 2: score += 50
                else: score = -1

            elif card.id == Fezandipiti_ex:
                if need_fezandipiti_draw or need_fezandipiti_for_setup:
                    score += 80 if not is_early else 30
                else: score = -1

            elif card.id == Genesect:
                if not op_used_ace_spec and (hand_counts[Lucky_Helmet] > 0 or hand_counts[Poke_Pad] > 0): score += 100
                else: score = -1

            elif card.id == Shaymin:
                score += 300 if op_has_water_threat else -1

            elif card.id == Lucky_Helmet: score = 7000

            elif card.id in RECOVERY_CARDS:
                if abra_line_on_field < 3: score = 10000
                else: score = -1

            elif card.id == Boss_Orders:
                score = 3200 if target_use_boss and target_can_kill else -1

            elif card.id == Hilda:
                score = 3000 if safe_draws >= 2 else -1

            elif card.id == Dawn:
                score = 3100 if safe_draws >= 3 else -1

            if bench_free <= 1 and score > 0 and card_table[card.id].cardType == CardType.POKEMON:
                score -= 5000

        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)

            if card.id == Lucky_Helmet:
                score = 7000
                if pokemon.id == Genesect and not op_used_ace_spec: score += 300
                elif o.inPlayArea == AreaType.ACTIVE: score += 200

            elif card.id in PSYCHIC_ENERGY_IDS:
                if len(pokemon.energyCards) >= 1: score = -1
                elif pokemon.id in ABRA_LINE:
                    score = 8000
                    if pokemon.id == Alakazam: score += 30
                    elif pokemon.id == Kadabra: score += 20
                    elif pokemon.id == Abra: score += 10
                    if o.inPlayArea == AreaType.ACTIVE: score += 5
                else: score = -1

            elif card.id == Enriching_Energy:
                if len(pokemon.energyCards) >= 1: score = -1
                elif pokemon.id in DUNSPARCE_LINE:
                    score = 8500
                    if pokemon.id == Dudunsparce: score += 10
                else: score = -1

        elif o.type == OptionType.EVOLVE:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000
            if card.id == Alakazam:
                if safe_draws < 3: score = -1
                elif o.inPlayArea == AreaType.ACTIVE: score += 200
                else: score += 50
            elif card.id == Kadabra:
                if safe_draws < 2: score = -1
                else:
                    score += 100 if len(pokemon.energies) == 0 else -20
                    if hand_counts[Rare_Candy] > 0 and hand_counts[Alakazam] > 0: score -= 100
            elif card.id == Dudunsparce:
                score += 80 if safe_draws >= 2 else -1

        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None:
                if card.id == Dudunsparce:
                    score = 30000 if need_dudunsparce_draw and safe_draws >= 3 else -1
                elif card.id == Fezandipiti_ex:
                    score = 29000 if (need_fezandipiti_draw or need_fezandipiti_for_setup) and safe_draws >= 3 else -1
                else:
                    score = 28000

        elif o.type == OptionType.RETREAT:
            if active_id == Alakazam and active_has_psychic: score = -1
            elif use_kadabra_finish and active_id != Kadabra and field_counts[Kadabra] >= 1: score = 2500
            elif active_id in (Abra, Dunsparce, Dudunsparce, Shaymin, Genesect):
                score = 2000 if field_counts[Alakazam] >= 1 or field_counts[Kadabra] >= 1 else -1
            else: score = -1

        elif o.type == OptionType.ATTACK:
            score = 1000
            if o.attackId == ATTACK_POWERFUL_HAND: score += 500
            elif o.attackId == ATTACK_SUPER_PSY_BOLT:
                score += 600 if op_active_hp <= 30 else 100
            elif o.attackId == ATTACK_TELEPORTATION: score += 50

        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    if context == SelectContext.MAIN and len(desc_indices) > 0:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None:
                if card.id == Dudunsparce: ability_used_dudunsparce = True
                elif card.id == Fezandipiti_ex: ability_used_fezandipiti = True

    return desc_indices[:select.maxCount]
