# TODO
# Switch Ice Cream and Potion from not only working on actives
# More In Depth Logic for Cards not already Done (Basically remove generic play score)
# Make Boss Order's Logic Better (Lowkey just look at someone else's code for this)
# Make Lillie's Logic Better
# Mist Energy Logic
# Xero Logic
# Ultra Ball Logic
# Logic for when to put energy on Kangaskhan(play around it)


from ptcg.api import (
    Observation,
    to_observation_class,
    OptionType,
    SelectContext,
    AreaType,
    CardType,
    Pokemon,
    Card,
    State,
    PlayerState,
    SelectData,
)

# Card IDs
BASIC_GRASS_ENERGY = 1
MIST_ENERGY = 11
GROW_GRASS_ENERGY = 18
DWEBBLE = 344
CRUSTLE = 345
MEGA_KANGASKHAN_EX = 756
ULTRA_BALL = 1121
POKEGEAR_3 = 1122
JUMBO_ICE_CREAM = 1147
BOSSS_ORDERS = 1182
TEAM_ROCKETS_PETREL = 1219
LILLIES_DETERMINATION = 1227
BUDDY_BUDDY_POFFIN = 1086
SUPER_POTION = 1112
HILDA = 1225
SWITCH = 1123
HEROS_CAPE = 1159
MORTYS_CONVICTION = 1187
XEROSICS_MACHINATIONS = 1197
TEAM_ROCKETS_FACTORY = 1257

CRUSTLE_SWITCH_IN_ENERGY_THRESHOLD = 3

# Deck loading
import os as _os

# __file__ is heuristics/crustle_agent/crustle_agent.py, so deck.csv (at the
# project root, alongside the other training scripts) is two directories up.
_project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_deck_path = _os.path.join(_project_root, "deck.csv")
with open(_deck_path) as _f:
    deck: list[int] = [int(line) for line in _f.readlines() if line.strip()]


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py if used."""
    global deck
    deck = deck_list


def _get_card(
    obs: Observation, area: AreaType, index: int, player_index: int
) -> Pokemon | Card | None:
    current = obs.current
    if current is None or index is None or area is None:
        return None

    if area == AreaType.DECK:
        if obs.select and obs.select.deck and 0 <= index < len(obs.select.deck):
            return obs.select.deck[index]
        return None

    if player_index is None or not (0 <= player_index < len(current.players)):
        return None
    ps: PlayerState = current.players[player_index]

    if area == AreaType.HAND:
        if ps.hand is not None and 0 <= index < len(ps.hand):
            return ps.hand[index]
        return None
    if area == AreaType.DISCARD:
        if 0 <= index < len(ps.discard):
            return ps.discard[index]
        return None
    if area == AreaType.ACTIVE:
        if 0 <= index < len(ps.active):
            return ps.active[index]
        return None
    if area == AreaType.BENCH:
        if 0 <= index < len(ps.bench):
            return ps.bench[index]
        return None
    if area == AreaType.PRIZE:
        if 0 <= index < len(ps.prize):
            return ps.prize[index]
        return None
    if area == AreaType.STADIUM:
        if 0 <= index < len(current.stadium):
            return current.stadium[index]
        return None
    if area == AreaType.LOOKING:
        if current.looking and 0 <= index < len(current.looking):
            return current.looking[index]
        return None
    return None


def _stadium_id(current: State) -> int | None:
    return current.stadium[0].id if current.stadium else None


def _factory_in_play(current: State) -> bool:
    return _stadium_id(current) == TEAM_ROCKETS_FACTORY


def _best_benched_crustle(ps: PlayerState) -> tuple[int | None, int]:
    """Return (bench_index, energy_count) for the highest-energy benched Crustle."""
    best_i, best_e = None, -1
    for i, mon in enumerate(ps.bench):
        if mon and mon.id == CRUSTLE:
            e = len(mon.energies)
            if e > best_e:
                best_i, best_e = i, e
    return best_i, max(best_e, 0)


def _crustle_ready(ps: PlayerState) -> bool:
    _, e = _best_benched_crustle(ps)
    return e >= CRUSTLE_SWITCH_IN_ENERGY_THRESHOLD


def _opponent_has_bench(current: State, your_index: int) -> bool:
    return len(current.players[1 - your_index].bench) > 0


def _active_id(ps: PlayerState) -> int | None:
    return ps.active[0].id if ps.active and ps.active[0] else None


def _active_pokemon(ps: PlayerState) -> Pokemon | None:
    return ps.active[0] if ps.active and ps.active[0] else None


def _crustle_line_count(ps: PlayerState) -> int:
    """Count Dwebble + Crustle across active and bench on our side."""
    count = 0
    for mon in ps.active or []:
        if mon and mon.id in (DWEBBLE, CRUSTLE):
            count += 1
    for mon in ps.bench or []:
        if mon and mon.id in (DWEBBLE, CRUSTLE):
            count += 1
    return count


def agent(obs_dict: dict) -> list[int]:
    """
    Priority order within a main-turn select (highest score wins):
      Switch + Hero's Cape          2200 / 2100 / 2050
      Boss's Orders (with target)   1700
      Hilda (when needed)           1600
      Jumbo Ice Cream (when useful) 1500  [hp < maxHp AND 3+ energy]
      Morty's Conviction            1400
      Kangaskhan Ability (no Crustle ready)  1100
      Energy attachment (to not-ready Crustle)  1050
      Generic attachment            1000
      Retreat (Kangaskhan -> Crustle swap)  950
      Evolve Dwebble                850
      Generic evolve                800
      Generic play                  600
      Crustle attack                150
      Generic attack                100
      Kangaskhan Ability (Crustle ready, de-prioritised)  300
      OptionType.END                -2  (safe baseline)
      Unmotivated retreat           -3
      Suppressed PLAY/ATTACH        -10
    """
    select = obs_dict.get("select")
    if select is None:
        return deck

    # Convert raw dict -> typed dataclasses once at the top.
    obs: Observation = to_observation_class(obs_dict)
    sel: SelectData = obs.select

    options = sel.option
    if not options:
        return []

    context: SelectContext = sel.context
    current: State | None = obs.current
    your_index: int | None = current.yourIndex if current else None

    min_count = sel.minCount
    max_count = min(sel.maxCount, len(options))
    min_count = min(min_count, max_count)

    your_ps: PlayerState | None = None
    opp_ps: PlayerState | None = None
    factory: bool = False
    crustle_rdy: bool = False

    if current is not None and your_index is not None:
        your_ps = current.players[your_index]
        opp_ps = current.players[1 - your_index]
        factory = _factory_in_play(current)
        crustle_rdy = _crustle_ready(your_ps)

    scores: list[int] = []

    for o in options:
        score = 0
        if context == SelectContext.MAIN and your_ps is not None:
            active_id = _active_id(your_ps)
            active_mon = _active_pokemon(your_ps)

            if o.type == OptionType.ABILITY:
                # Kangaskhan's Ability: use every turn until Crustle is ready.
                if active_id == MEGA_KANGASKHAN_EX:
                    score = 3000

            elif o.type == OptionType.RETREAT:
                # Only retreat when we want to swap Kangaskhan out for Crustle.
                if active_id == MEGA_KANGASKHAN_EX and crustle_rdy:
                    score = 950
                else:
                    score = -3

            elif o.type == OptionType.ATTACH:
                MAX_ENERGY = 4
                target_energy = 0
                score = 1000
                card = _get_card(obs, o.area, o.index, your_index)
                if o.inPlayArea == AreaType.ACTIVE and your_ps.active:
                    target_mon = your_ps.active[0] if your_ps.active else None
                    if target_mon:
                        target_energy = len(target_mon.energies)
                elif o.inPlayArea == AreaType.BENCH:
                    bench = your_ps.bench
                    idx = o.inPlayIndex
                    if idx is not None and 0 <= idx < len(bench) and bench[idx]:
                        target_energy = len(bench[idx].energies)
                if card is not None and card.id == HEROS_CAPE:
                    # Hero's Cape: prioritise active Crustle, then benched Crustle.
                    if o.inPlayArea == AreaType.ACTIVE and active_id == CRUSTLE:
                        score = 2100
                    elif o.inPlayArea == AreaType.BENCH:
                        bench_i, _ = _best_benched_crustle(your_ps)
                        if bench_i is not None and o.inPlayIndex == bench_i:
                            score = 2050
                        else:
                            score = -10  # don't waste Cape on a non-Crustle bench slot
                    else:
                        score = -10
                elif target_energy >= MAX_ENERGY:
                    score = -10
                else:
                    # Normal energy/tool: prefer building up the not-yet-ready Crustle.
                    bench_i, bench_e = _best_benched_crustle(your_ps)
                    if (
                        bench_i is not None
                        and bench_e < CRUSTLE_SWITCH_IN_ENERGY_THRESHOLD
                        and o.inPlayArea == AreaType.BENCH
                        and o.inPlayIndex == bench_i
                    ):
                        score = 1050

            elif o.type == OptionType.EVOLVE:
                card = _get_card(obs, o.area, o.index, your_index)
                score = (
                    850
                    if (
                        card is not None
                        and card.id == DWEBBLE
                        and _crustle_line_count(your_ps) < 2
                    )
                    else 0
                )

            elif o.type == OptionType.PLAY:
                score = 760
                card = _get_card(obs, AreaType.HAND, o.index, your_index)
                if card is not None:
                    cid = card.id

                    if cid == SWITCH and crustle_rdy and active_id != CRUSTLE:
                        score = 2200

                    elif cid == BOSSS_ORDERS:
                        score = (
                            1700 if _opponent_has_bench(current, your_index) else -10
                        )

                    elif cid == HILDA:
                        # Prioritise when we need Crustle or Grow Grass Energy.
                        bench_i, bench_e = _best_benched_crustle(your_ps)
                        need_crustle = active_id != CRUSTLE and bench_i is None
                        need_energy = bench_i is not None and not crustle_rdy
                        score = 1600 if (need_crustle or need_energy) else 700

                    elif cid == JUMBO_ICE_CREAM:
                        # Heal only when active is actually damaged and has 3+ energies.
                        mon = active_mon
                        if (
                            mon is not None
                            and mon.hp < mon.maxHp - 30
                            and len(mon.energies) >= 3
                        ):
                            score = 1500
                        else:
                            score = 0

                    elif cid == SUPER_POTION:
                        # Only worth playing when damaged.
                        mon = active_mon
                        score = (
                            1300 if (mon is not None and mon.hp < mon.maxHp - 30) else 0
                        )

                    elif cid == MORTYS_CONVICTION:
                        score = 1400

                    elif cid == TEAM_ROCKETS_PETREL:
                        if not factory:
                            score = 200
                        if crustle_rdy and active_id != CRUSTLE:
                            # Crustle is ready to switch from bench to active, but no switch in hand. Petrel can retrieve switch.
                            switch_in_hand = any(
                                c.id == SWITCH
                                for c in (your_ps.hand or [])
                                if c is not None
                            )
                            score = 2150 if not switch_in_hand else 800
                    elif cid == TEAM_ROCKETS_FACTORY:
                        score = 1100 if not factory else -10

                    elif cid == LILLIES_DETERMINATION:
                        score = 0

            elif o.type == OptionType.ATTACK:
                # Attack last (ends turn); Crustle attacking is the goal.
                score = 750

            elif o.type == OptionType.END:
                score = 700
        else:
            score = 2000

            if o.type == OptionType.CARD:
                card = _get_card(obs, o.area, o.index, o.playerIndex or your_index)

                # Switch/TO_ACTIVE: prefer a ready benched Crustle.
                if (
                    context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE)
                    and your_ps
                ):
                    bench_i, _ = _best_benched_crustle(your_ps)
                    if (
                        bench_i is not None
                        and o.area == AreaType.BENCH
                        and o.index == bench_i
                    ):
                        score += 1000

                # TODO Boss's target, check if can force kill target + win off prizes
                if context == SelectContext.EFFECT_TARGET and your_index is not None:
                    if o.playerIndex == 1 - your_index:
                        score += 500 if o.area == AreaType.ACTIVE else 800
                        if isinstance(card, Pokemon):
                            score += len(card.energies) * 50

                # Prefer Crustle line and Grow Grass Energy for evolving, and energies in hand
                if context in (
                    SelectContext.EVOLVE,
                    SelectContext.EVOLVES_FROM,
                    SelectContext.EVOLVES_TO,
                    SelectContext.TO_BENCH,
                    SelectContext.TO_FIELD,
                    SelectContext.TO_HAND,
                ):
                    if card is not None:
                        if card.id in (DWEBBLE, CRUSTLE, TEAM_ROCKETS_PETREL):
                            score += 600
                        elif card.id == HILDA:
                            score += 500
                        elif card.id == GROW_GRASS_ENERGY:
                            score += 400

                if card is not None and isinstance(card, Pokemon):
                    if your_index is not None and o.playerIndex == 1 - your_index:
                        score += 500 if o.area == AreaType.ACTIVE else 100
                        score += len(card.energies) * 50
                    else:
                        score += card.hp

            elif o.type == OptionType.YES:
                score += 100

            elif o.type == OptionType.NUMBER:
                score += o.number or 0

        scores.append(score)

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    output: list[int] = []
    for i in range(min(len(sorted_indices), max_count)):
        idx = sorted_indices[i]
        if scores[idx] >= 0 or len(output) < min_count:
            output.append(idx)

    return output
