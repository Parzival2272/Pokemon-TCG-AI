"""Mega Starmie ex / Mega Froslass ex — Rule-based agent

Deck Concept:
  Turn 1: start Staryu Active, bench Dunsparce + 2-3 attackers (Staryu/Snorunt)
  with Buddy-Buddy Poffin / Poke Pad. Turn 2: evolve into Mega Starmie ex and
  start sniping evolving Basics with Jetting Blow ({W}=120 + 50 to a Benched
  Pokemon). Against high-HP targets (and ALWAYS against Crustle's wall),
  attach Ignition Energy and fire Nebula Beam ({W}{W}{W}=210, ignores
  Weakness/Resistance/effects), or attack with Mega Froslass ex instead.
  Dudunsparce's Run Away Draw keeps the hand flowing (draw 3, shuffle itself
  back in -> re-bench Dunsparce later with saved Poffins).

Pokemon:
  Staryu             x4  Basic {W} HP70.  Water Gun {W}=20.
  Mega Starmie ex    x3  HP330. Jetting Blow {W}=120 (+50 to 1 opp Bench).
                         Nebula Beam {W}{W}{W}=210 (ignores W/R and effects).
  Snorunt            x2  Basic {W} HP60.  Astonish {W}{C}=20 (hand disruption).
  Mega Froslass ex   x2  HP310. Resentful Refrain {W}=50x. Absolute Snow
                         {W}{C}{C}=150 (opp Active now Asleep).
  Dunsparce          x2  Basic {C} HP60.
  Dudunsparce        x2  Stage 1 HP140. Ability Run Away Draw: draw 3, then
                         shuffle this Pokemon + attachments into the deck.

Trainers:
  Wally's Compassion x4 (heal all damage from 1 Mega, energy -> hand)
  Lillie's Determination x4, Hilda x3 (search Evolution Pokemon + Energy),
  Boss's Orders x2, Buddy-Buddy Poffin x4, Pokegear 3.0 x4, Poke Pad x4,
  Crushing Hammer x4, Ultra Ball x3, Mega Signal x1, Risky Ruins x4.

Energy: Basic Water x5, Ignition Energy x3 (discards at end of turn).

Strategy rules implemented (from deck guidelines):
  * Setup: Staryu Active > Snorunt > Dunsparce.
  * Bench plan: 2-3 attackers (3 = sweet spot, never more than 4) + 2 Dunsparce.
  * Always bench Dunsparce / evolve Dudunsparce, use Run Away Draw liberally.
  * Crushing Hammer ASAP: special energy on Active > basic on Active > Bench.
  * Poffin for basics; Poke Pad also fetches Dudunsparce; Ultra Ball / Mega
    Signal prioritize the Mega that evolves from our Active Pokemon.
  * Ultra Ball discards: spread across card types, save Boss/Wally/Megas.
  * Risky Ruins whenever possible.
  * Attacker choice by matchup: against evolving decks (small basics that
    want to evolve) prioritize Mega Starmie ex; against bigger basic decks
    or fully-evolved boards prioritize Mega Froslass ex (Resentful Refrain
    = 50 x cards in the opponent's hand; Absolute Snow 150 + Sleep).
  * Supporters: Lillie for setup, Wally when a Mega is heavily damaged, Boss
    for KO-able multi-prize bench targets, Hilda for Mega + Energy (Ignition
    when Nebula Beam is needed) or under Item lock. Pokegear digs for the
    supporter we currently want.
  * Energy: Water on attackers; Ignition on the Active Mega only on a turn we
    want to Nebula Beam (it self-discards, so don't waste it).

Score system (MAIN phase — higher score happens first in the turn):
  evolve 90000 > stadium 80000 > Run Away Draw 70000 > bench basics 60000 >
  Crushing Hammer 55000 > search items ~43-50k > supporter 35000 >
  attach ~20000 > retreat ~12000 > attack (ends turn) 1000+dmg > end 0.
  Negative scores are skipped whenever minCount allows.
"""

import os
import random
import sys
from collections import defaultdict

try:
    ROOT = __file__
except NameError:
    ROOT = None
CG_PATH = "/kaggle_simulations/agent"
for p in ([os.path.dirname(os.path.abspath(ROOT))] if ROOT else []) + [CG_PATH]:
    if p and p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

from ptcg.api import (
    AreaType,
    CardType,
    LogType,
    OptionType,
    SelectContext,
    all_card_data,
    to_observation_class,
)

try:
    from ptcg.api import all_attack
    ALL_ATTACKS = {a.attackId: a for a in all_attack()}
except Exception:
    ALL_ATTACKS = {}

# ── Deck loading ──

def read_deck_csv():
    fp = os.path.join(os.path.dirname(os.path.abspath(ROOT)), "deck.csv") if ROOT else "deck.csv"
    if not os.path.exists(fp):
        fp = "/kaggle_simulations/agent/deck.csv"
    with open(fp) as f:
        return [int(line) for line in f.read().splitlines() if line.strip()]


my_deck: list[int] = read_deck_csv()


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py."""
    global my_deck
    my_deck = list(deck_list)
    _resolve_ids()


CARD_DB = {c.cardId: c for c in all_card_data()}


# ── Card IDs — verified against Card_ID_List_EN.pdf:
#   1030 Staryu (POR 20)          1031 Mega Starmie ex (POR 21)
#    103 Snorunt (TWM 51)          861 Mega Froslass ex (ASC 47)
#     65 Dunsparce (TEF 128)        66 Dudunsparce (TEF 129, Run Away Draw)
#   1229 Wally's Compassion       1227 Lillie's Determination
#   1225 Hilda                    1182 Boss's Orders
#   1086 Buddy-Buddy Poffin       1122 Pokegear 3.0     1152 Poke Pad
#   1120 Crushing Hammer          1121 Ultra Ball       1145 Mega Signal
#   1260 Risky Ruins                 3 Basic {W} Energy   17 Ignition Energy

STARYU = STARMIE = SNORUNT = FROSLASS = DUNSPARCE = DUDUNSPARCE = 0
WALLY = LILLIE = HILDA = BOSS = 0
POFFIN = POKEGEAR = POKE_PAD = HAMMER = ULTRA_BALL = MEGA_SIGNAL = 0
RISKY_RUINS = WATER_ENERGY = IGNITION = 0

# Attack IDs (resolved from card data attack lists, in printed order)
WATER_GUN = JETTING_BLOW = NEBULA_BEAM = ASTONISH = 0
RESENTFUL_REFRAIN = ABSOLUTE_SNOW = LAND_CRUSH = 0

_ATTACK_BASE_DMG: dict[int, int] = {}


def _pick(primary: int, *alternates: int) -> int:
    """Use the printing that is actually in deck.csv, if an alternate is."""
    for cid in (primary,) + alternates:
        if cid in my_deck:
            return cid
    return primary


def _attacks_of(card_id: int) -> list[int]:
    data = CARD_DB.get(card_id)
    return list(getattr(data, "attacks", None) or []) if data else []


def _nth_attack(card_id: int, n: int) -> int:
    atks = _attacks_of(card_id)
    return atks[n] if n < len(atks) else 0


def _resolve_ids():
    global STARYU, STARMIE, SNORUNT, FROSLASS, DUNSPARCE, DUDUNSPARCE
    global WALLY, LILLIE, HILDA, BOSS
    global POFFIN, POKEGEAR, POKE_PAD, HAMMER, ULTRA_BALL, MEGA_SIGNAL
    global RISKY_RUINS, WATER_ENERGY, IGNITION
    global WATER_GUN, JETTING_BLOW, NEBULA_BEAM, ASTONISH
    global RESENTFUL_REFRAIN, ABSOLUTE_SNOW, LAND_CRUSH, _ATTACK_BASE_DMG

    STARYU = _pick(1030, 360)          # Staryu (Misty's Staryu alt)
    STARMIE = 1031                     # Mega Starmie ex
    SNORUNT = _pick(103, 860)          # Snorunt TWM (ASC alt)
    FROSLASS = 861                     # Mega Froslass ex
    DUNSPARCE = _pick(65, 305, 996)    # Dunsparce TEF (JTG / Larry's alts)
    DUDUNSPARCE = _pick(66, 306, 997)  # Dudunsparce TEF (ex alts)
    WALLY = 1229                       # Wally's Compassion
    LILLIE = 1227                      # Lillie's Determination
    HILDA = 1225                       # Hilda
    BOSS = 1182                        # Boss's Orders
    POFFIN = 1086                      # Buddy-Buddy Poffin
    POKEGEAR = 1122                    # Pokegear 3.0
    POKE_PAD = 1152                    # Poke Pad
    HAMMER = 1120                      # Crushing Hammer
    ULTRA_BALL = 1121                  # Ultra Ball
    MEGA_SIGNAL = 1145                 # Mega Signal
    RISKY_RUINS = 1260                 # Risky Ruins
    WATER_ENERGY = 3                   # Basic {W} Energy
    IGNITION = 17                      # Ignition Energy

    WATER_GUN = _nth_attack(STARYU, 0)
    JETTING_BLOW = _nth_attack(STARMIE, 0)
    NEBULA_BEAM = _nth_attack(STARMIE, 1)
    ASTONISH = _nth_attack(SNORUNT, 0)
    RESENTFUL_REFRAIN = _nth_attack(FROSLASS, 0)
    ABSOLUTE_SNOW = _nth_attack(FROSLASS, 1)
    LAND_CRUSH = _nth_attack(DUDUNSPARCE, 0)

    _ATTACK_BASE_DMG = {
        WATER_GUN: 20, JETTING_BLOW: 120, NEBULA_BEAM: 210, ASTONISH: 20,
        RESENTFUL_REFRAIN: 50, ABSOLUTE_SNOW: 150, LAND_CRUSH: 90,
    }
    _ATTACK_BASE_DMG.pop(0, None)


_resolve_ids()

# Opponent walls: Crustle line prevents damage from Rule Box Pokemon.
CRUSTLE_LINE = {344, 345, 532}
CRUSTLE_WALL = 345

ATTACKER_BASICS = lambda: {STARYU, SNORUNT}
MEGAS = lambda: {STARMIE, FROSLASS}
MEGA_FOR = lambda: {STARYU: STARMIE, SNORUNT: FROSLASS}


# ── Log tracking (Item lock detection) ──

_cur_turn_logs = []
_item_locked = False


def _update_log_tracking(obs):
    global _item_locked, _cur_turn_logs
    yi = obs.current.yourIndex
    for entry in obs.logs:
        if entry.type == LogType.TURN_END:
            _item_locked = any(
                prev.type == LogType.ATTACK
                and getattr(prev, "playerIndex", yi) != yi
                and prev.attackId == 323  # Itchy Pollen
                for prev in _cur_turn_logs)
            _cur_turn_logs.clear()
        else:
            _cur_turn_logs.append(entry)


# ── Board helpers ──

def get_card(obs, area, index, player_index):
    if area is None or index is None:
        return None
    ps = obs.current.players[player_index]
    try:
        if area == AreaType.DECK and obs.select and obs.select.deck is not None:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING and obs.current.looking is not None:
            return obs.current.looking[index]
    except (IndexError, TypeError):
        return None
    return None


def option_card(obs, opt):
    yi = obs.current.yourIndex
    pi = opt.playerIndex if getattr(opt, "playerIndex", None) is not None else yi
    if opt.type == OptionType.PLAY:
        return get_card(obs, AreaType.HAND, opt.index, pi)
    return get_card(obs, opt.area, opt.index, pi)


def option_target(obs, opt):
    if getattr(opt, "inPlayArea", None) is None or getattr(opt, "inPlayIndex", None) is None:
        return None
    return get_card(obs, opt.inPlayArea, opt.inPlayIndex, obs.current.yourIndex)


def my_state(obs):
    return obs.current.players[obs.current.yourIndex]


def opp_state(obs):
    return obs.current.players[1 - obs.current.yourIndex]


def active_pokemon(obs):
    ps = my_state(obs)
    return ps.active[0] if ps.active else None


def opp_active_pokemon(obs):
    ps = opp_state(obs)
    return ps.active[0] if ps.active else None


def opp_bench_pokemon(obs):
    return [p for p in (opp_state(obs).bench or []) if p]


def all_my_pokemon(obs):
    ps = my_state(obs)
    return [p for p in (list(ps.active or []) + list(ps.bench or [])) if p]


def hand_ids(obs):
    return [c.id for c in (my_state(obs).hand or []) if c]


def energy_count(pokemon):
    if pokemon is None:
        return 0
    if getattr(pokemon, "energyCards", None) is not None:
        return len(pokemon.energyCards)
    return len(getattr(pokemon, "energies", []) or [])


def has_ignition(pokemon):
    return any(getattr(c, "id", 0) == IGNITION
               for c in (getattr(pokemon, "energyCards", None) or []))


def damage_on(pokemon):
    if pokemon is None:
        return 0
    return max(0, getattr(pokemon, "maxHp", pokemon.hp) - pokemon.hp)


def count_in_play(obs, card_id):
    return sum(1 for p in all_my_pokemon(obs) if p.id == card_id)


def attackers_in_play(obs):
    """Basics + their evolved Megas count toward the attacker-slot budget."""
    ids = ATTACKER_BASICS() | MEGAS()
    return sum(1 for p in all_my_pokemon(obs) if p.id in ids)


def dunsparce_in_play(obs):
    return sum(1 for p in all_my_pokemon(obs) if p.id in (DUNSPARCE, DUDUNSPARCE))


def prize_value(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def is_rule_box(card_id):
    data = CARD_DB.get(card_id)
    return bool(data and (getattr(data, "ex", False) or getattr(data, "megaEx", False)))


def is_water_weak(pokemon):
    if pokemon is None:
        return False
    data = CARD_DB.get(pokemon.id)
    w = getattr(data, "weakness", None) if data else None
    if w is None:
        return False
    return getattr(w, "value", w) == WATER_ENERGY


def card_type(card_id):
    data = CARD_DB.get(card_id)
    return getattr(data, "cardType", None) if data else None


def deck_counts(obs):
    """Estimated copies of each card still in deck+prizes (decklist minus seen zones)."""
    counts = defaultdict(int)
    for cid in my_deck:
        counts[cid] += 1

    def sub(card):
        if card is None:
            return
        counts[card.id] -= 1
        for c in (getattr(card, "energyCards", None) or []):
            counts[c.id] -= 1
        for c in (getattr(card, "tools", None) or []):
            counts[c.id] -= 1
        for c in (getattr(card, "preEvolution", None) or []):
            counts[c.id] -= 1

    ps = my_state(obs)
    for zone in (ps.hand, ps.discard, ps.bench, ps.active):
        for card in (zone or []):
            sub(card)
    for card in (obs.current.stadium or []):
        if card is not None and getattr(card, "playerIndex", obs.current.yourIndex) == obs.current.yourIndex:
            counts[card.id] -= 1
    return counts


# ── Game plan helpers ──

def detect_crustle(obs):
    opp = opp_state(obs)
    ids = {p.id for p in (list(opp.active or []) + list(opp.bench or [])) if p}
    return bool(ids & CRUSTLE_LINE)


def need_nebula(obs):
    """Do we need the Nebula Beam plan (Ignition on Active)?
    Yes vs Crustle (wall ignores Rule Box damage), or vs a fat Active that
    Jetting Blow can't KO (weakness-adjusted)."""
    if detect_crustle(obs):
        return True
    opp = opp_active_pokemon(obs)
    if opp is None:
        return False
    jetting = 240 if is_water_weak(opp) else 120
    return opp.hp > jetting


def preferred_mega(obs):
    """Matchup-based attacker preference.
    Against evolving decks (boards full of small evolving basics) prioritize
    Mega Starmie ex (Jetting Blow snipes the evolving basics on the bench).
    Against bigger basic decks or fully-evolved boards prioritize
    Mega Froslass ex (better raw damage into high-HP targets)."""
    evolving = big = 0
    opp = opp_state(obs)
    for p in (list(opp.active or []) + list(opp.bench or [])):
        if p is None:
            continue
        data = CARD_DB.get(p.id)
        if data is None:
            continue
        max_hp = getattr(p, "maxHp", p.hp) or p.hp
        if getattr(data, "stage1", False) or getattr(data, "stage2", False) \
                or getattr(data, "megaEx", False):
            big += 1        # fully evolved Pokemon
        elif getattr(data, "ex", False) or max_hp >= 130:
            big += 1        # big basic (basic ex / tank)
        else:
            evolving += 1   # small basic that likely wants to evolve
    return FROSLASS if big > evolving else STARMIE


def mega_needed_id(obs):
    """Which Mega do we want in hand?
    Candidates are Megas whose basic is already in play; order by the
    matchup-preferred Mega first, then by evolving from our Active."""
    ids = hand_ids(obs)
    act = active_pokemon(obs)
    pref = preferred_mega(obs)
    cands = []  # (mega_id, from_active)
    if act and act.id in MEGA_FOR():
        cands.append((MEGA_FOR()[act.id], True))
    for p in (my_state(obs).bench or []):
        if p and p.id in MEGA_FOR():
            cands.append((MEGA_FOR()[p.id], False))
    cands.sort(key=lambda t: (t[0] == pref, t[1]), reverse=True)
    for w, _ in cands:
        if w not in ids and count_in_play(obs, w) < 2:
            return w
    return 0


def evolvable_basic(obs, basic_id):
    """Do we have a basic of this id in play that has sat a turn?"""
    return any(p.id == basic_id and not getattr(p, "appearThisTurn", True)
               for p in all_my_pokemon(obs))


def bench_ready_mega(obs):
    """A benched Mega with enough energy to Jetting Blow / attack now."""
    for i, p in enumerate(my_state(obs).bench or []):
        if p and p.id in MEGAS() and energy_count(p) >= 1:
            return i
    return -1


def water_in_hand(obs):
    return hand_ids(obs).count(WATER_ENERGY)


# ── Supporter planning ──

def desired_supporter(obs):
    """Pick the supporter we most want this turn (guideline priority).
    Returns (card_id, score)."""
    if obs.current.supporterPlayed:
        return 0, 0
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    best = (0, 0)

    # Wally: a Mega carrying a lot of damage -> full heal.
    worst_mega_dmg = max([damage_on(p) for p in all_my_pokemon(obs) if p.id in MEGAS()] or [0])
    if worst_mega_dmg >= 120:
        best = max(best, (WALLY, 30000 + worst_mega_dmg * 10), key=lambda x: x[1])

    # Boss: damaged multi-prize bench target we can KO easily.
    our_dmg = expected_attack_damage(obs)
    for target in opp_bench_pokemon(obs):
        if prize_value(target) >= 2 and damage_on(target) > 0 and target.hp <= our_dmg:
            best = max(best, (BOSS, 40000 + prize_value(target) * 1000), key=lambda x: x[1])
        elif target.hp <= our_dmg and prize_value(target) >= len(my_state(obs).prize):
            best = max(best, (BOSS, 60000), key=lambda x: x[1])  # lethal

    # Hilda: many basics down and we want a Mega + Energy to start attacking,
    # or we are Item-locked (searches without Items).
    mega_want = mega_needed_id(obs)
    energy_short = water_in_hand(obs) == 0 and IGNITION not in ids
    if (mega_want and (attackers_in_play(obs) >= 2) and (energy_short or _item_locked)) \
            or (_item_locked and mega_want):
        if dk[mega_want] > 0:
            best = max(best, (HILDA, 28000), key=lambda x: x[1])
    # Hilda also digs Ignition for the Nebula plan.
    if need_nebula(obs) and IGNITION not in ids and dk[IGNITION] > 0 \
            and any(p.id in MEGAS() for p in all_my_pokemon(obs)):
        best = max(best, (HILDA, 29000), key=lambda x: x[1])

    # Lillie: need attackers / bench building, or hand is gassed out.
    need_setup = attackers_in_play(obs) < 2 or dunsparce_in_play(obs) < 1 \
        or (mega_needed_id(obs) and not any(i in MEGAS() for i in ids))
    useful = sum(1 for i in ids if i not in (WALLY,))  # rough usefulness
    if need_setup or len(ids) <= 3:
        best = max(best, (LILLIE, 25000 - useful * 500), key=lambda x: x[1])

    return best


def refrain_damage(obs):
    """Resentful Refrain: 50 damage for each card in the opponent's hand."""
    return 50 * (getattr(opp_state(obs), "handCount", 0) or 0)


def expected_attack_damage(obs):
    """Rough best damage we can put out this turn (for Boss planning)."""
    act = active_pokemon(obs)
    best = 0
    for p in [act] + list(my_state(obs).bench or []):
        if p is None:
            continue
        e = energy_count(p)
        if p.id == STARMIE:
            if e >= 3 or has_ignition(p):
                best = max(best, 210)
            if e >= 1:
                best = max(best, 120)
        elif p.id == FROSLASS:
            if e >= 3 or has_ignition(p):
                best = max(best, 150)
            if e >= 1:
                best = max(best, refrain_damage(obs))
        elif p.id == STARYU and e >= 1:
            best = max(best, 20)
    return best


# ── Setup ──

_SETUP_ACTIVE_PRIORITY = {}


def score_setup(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ctx = obs.select.context

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        # Evolution deck: going second lets us attack turn 1 fundamentals sooner.
        return (10000, "choose second") if opt.type == OptionType.NO else (0, "go first")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        # Guideline: Staryu first > Snorunt > Dunsparce.
        pri = {STARYU: 100000, SNORUNT: 50000, DUNSPARCE: 10000}
        return pri.get(cid, 0), "setup Active"
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        # Bench during setup is free (no Risky Ruins counters yet):
        # always get Dunsparce down, then attackers up to plan.
        if cid == DUNSPARCE and count_in_play(obs, DUNSPARCE) < 2:
            return 50000, "setup bench Dunsparce"
        if cid == STARYU and attackers_in_play(obs) < 3:
            return 30000, "setup bench Staryu"
        if cid == SNORUNT and attackers_in_play(obs) < 3:
            return 20000, "setup bench Snorunt"
        return -1000, "hold"
    return 0, "non-setup"


# ── MAIN-phase scoring ──

def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    atk_in_play = attackers_in_play(obs)
    dun_in_play = dunsparce_in_play(obs)
    bench_used = len([p for p in (my_state(obs).bench or []) if p])

    # ── Basics from hand ──
    if cid == DUNSPARCE:
        # Always bench Dunsparce (target 2). Risky Ruins will ping it, accept it.
        if dun_in_play < 2:
            return 60000, "bench Dunsparce"
        return -500, "enough Dunsparce"
    if cid in (STARYU, SNORUNT):
        pref_basic = SNORUNT if preferred_mega(obs) == FROSLASS else STARYU
        if atk_in_play < 3:
            return (59000 if cid == pref_basic else 58000), "bench attacker"
        if atk_in_play < 4 and dk[STARMIE] + dk[FROSLASS] + ids.count(STARMIE) + ids.count(FROSLASS) > 0:
            return 20000, "bench 4th attacker"
        return -500, "attackers full"

    # ── Stadium ──
    if cid == RISKY_RUINS:
        stadium = [c for c in (obs.current.stadium or []) if c]
        if stadium and stadium[0].id == RISKY_RUINS:
            return -500, "Risky Ruins already up"
        return 80000, "play Risky Ruins (always)"

    # ── Items ──
    if cid == HAMMER:
        # ASAP whenever the opponent has any energy in play.
        opp_has_energy = any(energy_count(p) > 0
                             for p in [opp_active_pokemon(obs)] + opp_bench_pokemon(obs) if p)
        if opp_has_energy:
            return 55000, "Crushing Hammer ASAP"
        return -500, "no target for Hammer"

    if cid == POFFIN:
        # Build 2-3 attackers + 2 Dunsparce; save extras to re-bench Dunsparce
        # after Run Away Draw.
        want_dun = dun_in_play < 2 and dk[DUNSPARCE] > 0
        want_atk = atk_in_play < 3 and (dk[STARYU] + dk[SNORUNT]) > 0
        if bench_used >= 5:
            return -500, "bench full"
        if want_dun or want_atk:
            return 50000, "Poffin: build bench"
        return -500, "save Poffin for later Dunsparce"

    if cid == POKE_PAD:
        # Like Poffin (basics) but can also grab Dudunsparce. Prefer Poffin
        # for basics; Pad shines fetching Dudunsparce for draw power.
        want_dudun = (count_in_play(obs, DUNSPARCE) > 0
                      and DUDUNSPARCE not in ids and dk[DUDUNSPARCE] > 0)
        if want_dudun:
            return 48000, "Poke Pad: fetch Dudunsparce"
        want_basic = ((dun_in_play < 2 and dk[DUNSPARCE] > 0)
                      or (atk_in_play < 3 and dk[STARYU] + dk[SNORUNT] > 0))
        if want_basic and POFFIN not in ids:
            return 44000, "Poke Pad: fetch basic (no Poffin)"
        return -500, "save Poke Pad"

    if cid == ULTRA_BALL:
        # Can find Megas — prioritize those unless we really need basics.
        if len(ids) < 3:  # can't pay the 2-card discard cost
            return -1000, "can't afford Ultra Ball"
        mega_want = mega_needed_id(obs)
        if mega_want and dk[mega_want] > 0:
            return 46000, "Ultra Ball: fetch Mega"
        if atk_in_play + dun_in_play <= 1 and dk[STARYU] + dk[SNORUNT] + dk[DUNSPARCE] > 0:
            return 45000, "Ultra Ball: emergency basic"
        return -500, "save Ultra Ball"

    if cid == MEGA_SIGNAL:
        mega_want = mega_needed_id(obs)
        if mega_want and dk[mega_want] > 0:
            return 45000, "Mega Signal: fetch Mega"
        return -500, "save Mega Signal"

    if cid == POKEGEAR:
        # Dig for the supporter we currently want but don't hold.
        want, _ = desired_supporter(obs)
        if want and want not in ids:
            return 43000, "Pokegear: dig for supporter"
        return -300, "no supporter needed"

    # ── Supporters (play only the planned one) ──
    if cid in (LILLIE, HILDA, BOSS, WALLY):
        want, wscore = desired_supporter(obs)
        if cid != want:
            return -500, "not the planned supporter"
        if cid == LILLIE:
            # Lillie shuffles our hand: let attaches/items resolve first.
            return 15000, "play Lillie (after items/attach)"
        return 35000, "play planned supporter"

    # ── Energy played from hand is handled via ATTACH options; anything else: ──
    if cid in MEGAS():
        return -1000, "Megas evolve, not played"
    return 1000, "generic play"


def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    tid = target.id if target else None
    is_active = getattr(opt, "inPlayArea", None) == AreaType.ACTIVE

    if cid in MEGAS():
        s = 90000 + energy_count(target) * 10
        if is_active:
            s += 5000  # active Mega starts attacking immediately
        # Don't over-commit a 3rd Mega of the same kind (only 2-3 copies each).
        if count_in_play(obs, cid) >= 2:
            s = 5000
        # vs Crustle a wall of Megas without Nebula energy is useless — still
        # fine to evolve (Nebula Beam is the answer), so no penalty.
        return s, "evolve to Mega"
    if cid == DUDUNSPARCE and tid == DUNSPARCE:
        return 88000, "evolve Dudunsparce"
    return 10000 + energy_count(target), "generic evolve"


def score_ability(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    if cid == DUDUNSPARCE:
        # Run Away Draw: use it more often than not.
        ps = my_state(obs)
        if ps.deckCount <= 3:
            return -500, "deck too thin for Run Away Draw"
        if ps.handCount >= 8:
            return -300, "hand already full"
        if len(all_my_pokemon(obs)) <= 1:
            return -500, "only Pokemon in play"
        return 70000, "Run Away Draw"
    return 1, "generic ability"


def attach_target_score(obs, target, in_active, attach_id):
    """Score attaching `attach_id` energy onto `target`."""
    if target is None:
        return -500
    tid = target.id
    e = energy_count(target)
    nebula_plan = need_nebula(obs)

    pref = preferred_mega(obs)

    if attach_id == IGNITION:
        # Self-discarding: only worth it on the Active Mega on a big-attack turn.
        if in_active and tid == STARMIE and nebula_plan:
            return 30000
        if in_active and tid == FROSLASS and e < 3:
            # Enables Absolute Snow (150) immediately.
            return 22000 if pref == FROSLASS else 15000
        if in_active and tid in MEGAS() and nebula_plan:
            return 15000
        return -1000

    # Water energy
    if tid in (DUNSPARCE, DUDUNSPARCE):
        return -800  # never fuel the draw engine
    score = 20000
    if tid == STARMIE:
        score += 3000
    elif tid == FROSLASS:
        score += 2500
    elif tid == STARYU:
        score += 2000
    elif tid == SNORUNT:
        score += 1500
    if tid in MEGAS() and tid == pref:
        score += 700  # feed the matchup-preferred attacker first
    if tid in MEGA_FOR() and MEGA_FOR()[tid] == pref:
        score += 300  # ...and its pre-evolution
    if in_active:
        score += 1500  # guideline: active attacker first
    if e == 0:
        score += 1000  # spread: everyone wants their 1st energy (attacks cost {W})
    elif e >= 3:
        score -= 5000
    elif nebula_plan and tid == STARMIE:
        score += 800  # building toward WWW manually
    elif tid == FROSLASS and pref == FROSLASS:
        score += 500  # building toward Absolute Snow (WCC)
    else:
        score -= 1200  # 2nd+ energy is low value when attacks cost 1
    return score


def score_attach(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    if obs.current.energyAttached:
        return -1000, "already attached"
    in_active = getattr(opt, "inPlayArea", None) == AreaType.ACTIVE
    return attach_target_score(obs, target, in_active, cid), "attach energy"


def score_retreat(obs, opt):
    act = active_pokemon(obs)
    ready = bench_ready_mega(obs)
    if act is None:
        return -100, "no active"
    act_ready = act.id in MEGAS() and energy_count(act) >= 1
    if not act_ready and ready >= 0:
        return 12000, "retreat into ready Mega"
    if act.id in (DUNSPARCE, DUDUNSPARCE) and ready >= 0:
        return 12500, "retreat Dunsparce line"
    return -100, "hold position"


def score_attack(obs, opt):
    aid = getattr(opt, "attackId", None)
    act = active_pokemon(obs)
    opp = opp_active_pokemon(obs)
    base = _ATTACK_BASE_DMG.get(aid, 0)
    if base == 0 and aid in ALL_ATTACKS:
        base = getattr(ALL_ATTACKS[aid], "damage", 0) or 0

    if aid == RESENTFUL_REFRAIN:
        base = refrain_damage(obs)  # 50 per card in opponent's hand

    dmg = base
    if opp is not None:
        if aid == NEBULA_BEAM:
            dmg = 210  # ignores Weakness/Resistance/effects
        elif is_water_weak(opp):
            dmg = base * 2
        # Crustle wall: Rule Box attackers deal 0 except Nebula Beam.
        if opp.id == CRUSTLE_WALL and act and is_rule_box(act.id) and aid != NEBULA_BEAM:
            return -800, "Crustle walls this attack"

    pref = preferred_mega(obs)
    score = 1000 + dmg
    if opp is not None and dmg >= opp.hp:
        score += 5000 + prize_value(opp) * 1000  # take the KO
    if aid == JETTING_BLOW:
        score += 200  # bench snipe value
        if pref == STARMIE:
            score += 300  # evolving-deck matchup: keep sniping
        if need_nebula(obs) and act and (energy_count(act) >= 3 or has_ignition(act)):
            score -= 400  # prefer Nebula into fat targets
    if aid == NEBULA_BEAM and need_nebula(obs):
        score += 800
    if aid in (RESENTFUL_REFRAIN, ABSOLUTE_SNOW) and pref == FROSLASS:
        score += 300  # big-basic / fully-evolved matchup: Froslass shines
    if aid == ABSOLUTE_SNOW:
        score += 100  # sleep is nice tempo
    return score, "attack"


# ── Selection scoring (searches / discards / targets) ──

def score_to_hand(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    effect = getattr(obs.select, "effect", None)
    effect_id = effect.id if effect else None
    atk_in_play = attackers_in_play(obs)
    dun_in_play = dunsparce_in_play(obs)
    mega_want = mega_needed_id(obs)

    # Pokegear: take the supporter we planned for.
    if effect_id == POKEGEAR:
        want, _ = desired_supporter(obs)
        if cid == want:
            return 30000, "Gear: planned supporter"
        pri = {LILLIE: 20000, HILDA: 15000, BOSS: 12000, WALLY: 8000}
        return pri.get(cid, 0), "Gear: fallback supporter"

    # Hilda: an Evolution Pokemon + an Energy.
    if effect_id == HILDA:
        if cid == mega_want:
            return 30000, "Hilda: Mega for Active line"
        if cid in MEGAS():
            return 22000, "Hilda: other Mega"
        if cid == DUDUNSPARCE and count_in_play(obs, DUNSPARCE) > 0 and DUDUNSPARCE not in ids:
            return 15000, "Hilda: Dudunsparce"
        if cid == IGNITION:
            return (28000 if need_nebula(obs) else 5000), "Hilda: Ignition"
        if cid == WATER_ENERGY:
            return (24000 if not need_nebula(obs) else 18000), "Hilda: Water"
        return 1000, "Hilda: generic"

    # Mega Signal: only Megas — take the Active line's evolution.
    if effect_id == MEGA_SIGNAL:
        if cid == mega_want:
            return 30000, "Signal: Active-line Mega"
        if cid in MEGAS():
            return 20000, "Signal: Mega"
        return 1000, "Signal: generic"

    # Ultra Ball: Megas first, basics only when truly needed.
    if effect_id == ULTRA_BALL:
        if cid == mega_want:
            return 30000, "UB: Active-line Mega"
        if cid in MEGAS() and cid not in ids and count_in_play(obs, cid) < 2:
            return 24000, "UB: Mega"
        if cid == DUNSPARCE and dun_in_play < 2:
            return 18000, "UB: Dunsparce"
        if cid in (STARYU, SNORUNT) and atk_in_play < 3:
            return 16000, "UB: attacker basic"
        if cid == DUDUNSPARCE and count_in_play(obs, DUNSPARCE) > 0:
            return 14000, "UB: Dudunsparce"
        return 1000, "UB: generic"

    # Generic to-hand ranking.
    if cid == mega_want:
        return 26000, "take Active-line Mega"
    if cid in MEGAS():
        return 20000, "take Mega"
    if cid == DUNSPARCE and dun_in_play < 2:
        return 17000, "take Dunsparce"
    if cid in (STARYU, SNORUNT) and atk_in_play < 3:
        pref_basic = SNORUNT if preferred_mega(obs) == FROSLASS else STARYU
        return (15500 if cid == pref_basic else 15000), "take attacker"
    if cid == DUDUNSPARCE and count_in_play(obs, DUNSPARCE) > 0 and DUDUNSPARCE not in ids:
        return 14000, "take Dudunsparce"
    if cid == WATER_ENERGY and water_in_hand(obs) == 0:
        return 12000, "take Water Energy"
    if cid == IGNITION and need_nebula(obs):
        return 13000, "take Ignition"
    if cid in (LILLIE, HILDA, BOSS, WALLY):
        return 8000, "take supporter"
    return 1000, "generic take"


def score_to_bench(obs, opt):
    """Poffin / Poke Pad putting Basics straight onto the Bench."""
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    atk_in_play = attackers_in_play(obs)
    dun_in_play = dunsparce_in_play(obs)

    if cid == DUNSPARCE:
        if dun_in_play < 2:
            return 30000, "bench Dunsparce (always want 2)"
        return -500, "enough Dunsparce"
    if cid in (STARYU, SNORUNT):
        pref_basic = SNORUNT if preferred_mega(obs) == FROSLASS else STARYU
        if atk_in_play < 3:
            return (25000 if cid == pref_basic else 22000), "bench attacker"
        return -500, "attackers full (3 is the sweet spot)"
    return 1000, "generic bench"


# Discard value: LOWER value = happier to discard.
def _keep_value(obs, cid):
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    if cid in MEGAS():
        return 9000
    if cid == BOSS or cid == WALLY:
        return 8000  # guideline: save high-value supporters
    if cid == WATER_ENERGY:
        # only 5 in the deck — precious unless we're flush
        return 7000 if water_in_hand(obs) <= 1 else 3500
    if cid == IGNITION:
        return 6000 if need_nebula(obs) else 2500
    if cid == HILDA:
        return 5000
    if cid in (STARYU, SNORUNT, DUNSPARCE):
        return 4500 if dk[cid] == 0 else 3000
    if cid == DUDUNSPARCE:
        return 3500
    if cid == POFFIN:
        return 4000  # saved for re-benching Dunsparce
    if cid == LILLIE:
        return 2500
    if cid == RISKY_RUINS:
        # 4 copies, only ever need one at a time
        return 500 if ids.count(RISKY_RUINS) > 1 else 2000
    if cid in (POKEGEAR, POKE_PAD, HAMMER, ULTRA_BALL, MEGA_SIGNAL):
        return 2200
    return 2000


def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    keep = _keep_value(obs, cid)
    score = 10000 - keep
    if ids.count(cid) > 1:
        score += 4000  # duplicates of the same card go first
    return score, "discard"


def score_opp_energy(obs, opt):
    """Crushing Hammer / similar: special on Active > basic on Active > Bench."""
    card = None
    try:
        card = get_card(obs, opt.area, opt.index, opt.playerIndex)
    except Exception:
        pass
    on_active = getattr(opt, "area", None) == AreaType.ACTIVE
    is_special = bool(card and card_type(card.id) == CardType.SPECIAL_ENERGY)
    score = 1000
    score += 2000 if on_active else 0
    score += 1000 if is_special else 0
    return score, "hammer target"


def score_target(obs, opt):
    """SWITCH / TO_ACTIVE / HEAL / DAMAGE etc."""
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ctx = obs.select.context
    yi = obs.current.yourIndex
    pi = getattr(opt, "playerIndex", yi)

    if ctx == SelectContext.HEAL:
        # Wally: heal the most damaged Mega.
        if cid in MEGAS():
            return 20000 + damage_on(card), "heal Mega"
        return damage_on(card), "heal"

    if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE,
               SelectContext.SETUP_ACTIVE_POKEMON):
        if pi != yi and card is not None:
            # Boss's Orders: damaged multi-prize target we can KO.
            dmg = expected_attack_damage(obs)
            s = prize_value(card) * 1000 + energy_count(card) * 100
            if card.hp <= dmg:
                s += 20000 + damage_on(card)
            elif damage_on(card) > 0:
                s += 3000
            return s, "Boss target"
        # Our promotion order (matchup-preferred Mega leads).
        e = energy_count(card) if card else 0
        pri = {STARMIE: 19000, FROSLASS: 18000, STARYU: 10000,
               SNORUNT: 8000, DUDUNSPARCE: 3000, DUNSPARCE: 2000}
        s = pri.get(cid, 1000) + e * 2000
        if cid == preferred_mega(obs):
            s += 3000
        return s, "promote"

    if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
               SelectContext.DAMAGE_COUNTER_ANY):
        # Jetting Blow snipe: KO first, otherwise soften evolving basics.
        if card is None:
            return 0, "no target"
        hp = card.hp
        s = 1000 + prize_value(card) * 500 + energy_count(card) * 100
        if 0 < hp <= 50:
            s += 15000  # 50-damage snipe KO
        data = CARD_DB.get(cid)
        is_basic = data and not getattr(data, "stage1", False) and not getattr(data, "stage2", False)
        if is_basic and hp <= 80:
            s += 4000  # likely an evolving basic — soften it for next turn
        s -= hp  # prefer the squishiest
        return s, "snipe target"

    if ctx in (SelectContext.TO_FIELD, SelectContext.TO_BENCH):
        return score_to_bench(obs, opt)

    return 1000, "generic target"


# ── Dispatcher ──

def score_option(obs, opt):
    ctx = obs.select.context

    if ctx in (SelectContext.IS_FIRST, SelectContext.MULLIGAN,
               SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON):
        if opt.type in (OptionType.YES, OptionType.NO) or opt.type == OptionType.CARD:
            return score_setup(obs, opt)

    if opt.type in (OptionType.YES, OptionType.NO):
        return ((1, "yes") if opt.type == OptionType.YES else (0, "no"))

    if opt.type == OptionType.NUMBER:
        return (getattr(opt, "number", 0) or 0), "number (draw max)"

    if opt.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        yi = obs.current.yourIndex
        if getattr(opt, "playerIndex", yi) != yi:
            return score_opp_energy(obs, opt)
        # Discarding our own energy (retreat cost etc.): Ignition first.
        card = get_card(obs, opt.area, opt.index, yi)
        if card and card.id == IGNITION:
            return 2000, "pay with Ignition (dies anyway)"
        return 1000, "pay energy"

    if ctx == SelectContext.MAIN:
        if opt.type == OptionType.PLAY:
            return score_play(obs, opt)
        if opt.type == OptionType.EVOLVE:
            return score_evolve(obs, opt)
        if opt.type == OptionType.ABILITY:
            return score_ability(obs, opt)
        if opt.type == OptionType.ATTACH:
            return score_attach(obs, opt)
        if opt.type == OptionType.RETREAT:
            return score_retreat(obs, opt)
        if opt.type == OptionType.ATTACK:
            return score_attack(obs, opt)
        if opt.type == OptionType.END:
            return 0, "end turn"
        return 500, "generic MAIN"

    if ctx == SelectContext.TO_HAND:
        return score_to_hand(obs, opt)
    if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
        return score_discard(obs, opt)
    if ctx == SelectContext.ATTACK:
        return score_attack(obs, opt)
    if opt.type == OptionType.ATTACH:
        return score_attach(obs, opt)
    if opt.type == OptionType.CARD:
        return score_target(obs, opt)
    if opt.type == OptionType.END:
        return 0, "end"
    return 100, "fallback"


# ── Choose & Agent ──

def _is_trainer_type(ct):
    return ct in (CardType.SUPPORTER, CardType.ITEM, CardType.STADIUM,
                  getattr(CardType, "TOOL", CardType.ITEM))


def choose_options(obs):
    select = obs.select
    ctx = select.context
    scored = []
    for i, opt in enumerate(select.option):
        try:
            score, reason = score_option(obs, opt)
        except Exception as e:
            score, reason = -999999, f"error {type(e).__name__}: {e}"
        scored.append([score, i, reason])

    # Ultra Ball / forced hand discards: greedy pick with card-TYPE diversity
    # ("1 supporter/stadium/item, avoid multiples of the same type").
    if ctx in (SelectContext.DISCARD, SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
        selected = []
        pool = list(scored)
        picked_types = defaultdict(int)
        picked_ids = defaultdict(int)
        need = max(select.minCount, 0)
        cap = select.maxCount
        while pool and len(selected) < cap:
            def adj(entry):
                s, i, _ = entry
                card = option_card(obs, select.option[i])
                cid = card.id if card else None
                ct = card_type(cid) if cid else None
                s2 = s
                if ct is not None and picked_types[ct] > 0 and _is_trainer_type(ct):
                    s2 -= 3000 * picked_types[ct]  # diversify trainer types
                if cid is not None and picked_ids[cid] > 0:
                    s2 += 1500  # ...but doubling the exact same card is fine
                return s2
            pool.sort(key=adj, reverse=True)
            best = pool.pop(0)
            if adj(best) < 0 and len(selected) >= need:
                break
            selected.append(best[1])
            card = option_card(obs, select.option[best[1]])
            if card:
                picked_ids[card.id] += 1
                ct = card_type(card.id)
                if ct is not None:
                    picked_types[ct] += 1
        while len(selected) < need and pool:
            selected.append(pool.pop(0)[1])
        if len(selected) < select.minCount:
            leftovers = [i for _, i, _ in sorted(scored, reverse=True) if i not in selected]
            selected += leftovers[: select.minCount - len(selected)]
        return selected

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    selected = []
    for score, i, reason in scored:
        if len(selected) >= select.maxCount:
            break
        if score < 0 and len(selected) >= select.minCount:
            continue
        selected.append(i)
    if len(selected) < select.minCount:
        selected = [i for _, i, _ in scored[: select.minCount]]
    return selected


def agent(obs_dict):
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial call: return the 60-card decklist.
        global _cur_turn_logs, _item_locked
        _cur_turn_logs.clear()
        _item_locked = False
        return my_deck
    _update_log_tracking(obs)
    if not obs.select.option:
        return []
    try:
        return choose_options(obs)
    except Exception:
        n = len(obs.select.option)
        k = max(obs.select.minCount, min(obs.select.maxCount, 1))
        return random.sample(list(range(n)), min(k, n))
