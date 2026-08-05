"""Marnie's Grimmsnarl ex / Froslass / Munkidori — Rule-based agent

Deck Concept:
  Turn 1: start Budew Active (0 retreat, 0-cost attack) and Itchy Pollen to
  Item-lock the opponent while we build. Bench Marnie's Impidimp, Munkidori,
  and Snorunt underneath it. Turn 2+: get Marnie's Grimmsnarl ex online as
  fast as possible (Rare Candy off an Impidimp is the dream), pivot out of
  Budew for free, and start hitting with Shadow Bullet (180 + 30 to a Bench).
  Grimmsnarl ex powers ITSELF up: Punk Up pulls up to 5 Basic {D} from deck
  when it evolves from hand, but ONLY onto Marnie's Pokemon — so our manual
  energy attachment for the turn always goes to Munkidori instead.
  Froslass then puts a counter on every Ability Pokemon each Checkup (ours
  included), and Munkidori's Adrena-Brain launders that self-damage onto the
  opponent, 3 counters at a time, once per Munkidori per turn.

Pokemon:
  Marnie's Grimmsnarl ex (648) x3  Stage 2 {D} HP320. Ability Punk Up: on evolve
                         from hand, search deck for up to 5 Basic {D} Energy and
                         attach to your Marnie's Pokemon in any way.
                         {D}{D} Shadow Bullet 180 (+30 to 1 Benched Pokemon).
                         Weak {G}. Retreat 2.
  Marnie's Impidimp (646)     x4  Basic {D} HP70. {C} Filch (draw a card),
                                  {D} Corkscrew Punch 10.
  Marnie's Morgrem (647)      x2  Stage 1 {D} HP100. {D}{D} Corkscrew Punch 60.
  Snorunt (860)               x2  Basic {W} HP60.
  Froslass (104)              x2  Stage 1. Ability Freezing Shroud: during
                                  Pokemon Checkup, 1 damage counter on EACH
                                  Pokemon with an Ability (both players),
                                  except any Froslass.
  Budew (235)                 x2  Basic {G} HP30, Retreat 0.
                                  {0} Itchy Pollen 10 — opponent can't play
                                  Item cards during their next turn.
  Munkidori (112)             x4  Basic {P} HP110. Ability Adrena-Brain: once
                                  during your turn, if this Pokemon has any
                                  {D} Energy attached, move up to 3 damage
                                  counters from 1 of YOUR Pokemon to 1 of your
                                  opponent's Pokemon.

Trainers:
  Spikemuth Gym x4 (Stadium; once per turn each player may search their deck
    for a Marnie's Pokemon — our ONLY way to search out Grimmsnarl ex),
  Team Rocket's Petrel x4 (search deck for any Trainer),
  Poke Pad x4 (search a Pokemon WITHOUT a Rule Box — so no Grimmsnarl ex),
  Buddy-Buddy Poffin x2 (2 Basic Pokemon with 70 HP or less — Impidimp/Snorunt/
    Budew only, NOT Munkidori at 110 HP),
  Lillie's Determination x4, Judge x2, Morty's Conviction x1,
  Boss's Orders x2, Rare Candy x2, Crushing Hammer x4,
  Unfair Stamp x1 (ACE SPEC), Air Balloon x1 (Retreat -2).

Energy: 10 Basic {D}.

Strategy rules implemented (from deck guidelines):
  * Go second. Budew Active on setup; Itchy Pollen turn 1 for the Item lock.
  * Optimal board = 1 Froslass + 2 Grimmsnarl lines + 3 Munkidori (6 slots).
    Search items and Spikemuth Gym exist to assemble exactly that.
  * Rare Candy + Impidimp -> Grimmsnarl ex is the highest-priority line.
  * Every turn: attack with Grimmsnarl, attach {D} to a Munkidori, evolve
    toward Froslass.
  * Energy NEVER goes on Grimmsnarl by hand unless Punk Up whiffed — Munkidori
    needs {D} to switch Adrena-Brain on, and Punk Up can't reach it.
  * Punk Up distribution: Active Grimmsnarl to 2 first, then the 2nd
    Grimmsnarl line, then pre-load Morgrem/Impidimp.
  * Adrena-Brain: pull damage off Grimmsnarl first (keep the attacker alive),
    dump it onto whatever it can Knock Out, else the biggest prize.
  * Supporters: Boss for a reachable target > Judge when the OPPONENT's hand is
    big and ours isn't > Morty when OUR hand is big (it doesn't shuffle it away)
    and their Bench is wide > Petrel to build the board (Poffin for
    Snorunt/Impidimp, Spikemuth, or Hammer once we're set up) > Lillie to refill.
  * Unfair Stamp whenever it is legal to play at all.
  * Crushing Hammer whenever the opponent has Energy; Special Energy first.

Score system (MAIN phase — higher score happens first in the turn):
  Spikemuth ability 98000 > play Stadium 95000 > Rare Candy 92000 >
  evolve 90000 > bench basics 75000 > Petrel 72000 > search items 68000 >
  Crushing Hammer 65000 > Air Balloon 60000 > Unfair Stamp 55000 >
  Boss's Orders 45000 > attach Energy 22000 > Adrena-Brain 20000 >
  draw Supporter 18000 > retreat 12000 > attack (ends turn) 1000+dmg > end 0.
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
#    648 Marnie's Grimmsnarl ex (DRI 136)   646 Marnie's Impidimp (DRI 134)
#    647 Marnie's Morgrem (DRI 135)         860 Snorunt (ASC 46)
#    104 Froslass (TWM 53)                  235 Budew (PRE 4)
#    112 Munkidori (TWM 95 / PRE 44)
#   1152 Poke Pad (POR 81)                 1227 Lillie's Determination (MEG 119)
#   1213 Judge (SVI 176)                   1259 Spikemuth Gym (DRI 169)
#   1079 Rare Candy (SVI 191)              1174 Air Balloon (BLK 79)
#   1187 Morty's Conviction (TEF 155)      1080 Unfair Stamp (TWM 165)
#   1219 Team Rocket's Petrel (DRI 176)    1086 Buddy-Buddy Poffin (ASC 184)
#   1120 Crushing Hammer (POR 71)          1182 Boss's Orders (MEG 114)
#      7 Basic {D} Energy (MEE 7)

GRIMMSNARL = MORGREM = IMPIDIMP = SNORUNT = FROSLASS = BUDEW = MUNKIDORI = 0
POKE_PAD = LILLIE = JUDGE = MORTY = PETREL = BOSS = 0
SPIKEMUTH = RARE_CANDY = AIR_BALLOON = UNFAIR_STAMP = POFFIN = HAMMER = 0
DARK_ENERGY = 0

# Attack IDs (resolved from card data attack lists, in printed order)
SHADOW_BULLET = CORKSCREW_MORGREM = FILCH = CORKSCREW_IMP = 0
ITCHY_POLLEN = 323  # shared engine ID; re-resolved below if card data exposes it
MIND_BEND = 0

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
    global GRIMMSNARL, MORGREM, IMPIDIMP, SNORUNT, FROSLASS, BUDEW, MUNKIDORI
    global POKE_PAD, LILLIE, JUDGE, MORTY, PETREL, BOSS
    global SPIKEMUTH, RARE_CANDY, AIR_BALLOON, UNFAIR_STAMP, POFFIN, HAMMER
    global DARK_ENERGY
    global SHADOW_BULLET, CORKSCREW_MORGREM, FILCH, CORKSCREW_IMP
    global ITCHY_POLLEN, MIND_BEND, _ATTACK_BASE_DMG

    GRIMMSNARL = 648                   # Marnie's Grimmsnarl ex
    MORGREM = 647                      # Marnie's Morgrem
    IMPIDIMP = 646                     # Marnie's Impidimp
    SNORUNT = _pick(860, 103)          # Snorunt ASC (TWM alt)
    FROSLASS = 104                     # Froslass TWM 53
    BUDEW = 235                        # Budew PRE 4
    MUNKIDORI = 112                    # Munkidori
    POKE_PAD = 1152                    # Poke Pad
    LILLIE = 1227                      # Lillie's Determination
    JUDGE = 1213                       # Judge
    MORTY = 1187                       # Morty's Conviction
    PETREL = 1219                      # Team Rocket's Petrel
    BOSS = 1182                        # Boss's Orders
    SPIKEMUTH = 1259                   # Spikemuth Gym
    RARE_CANDY = 1079                  # Rare Candy
    AIR_BALLOON = 1174                 # Air Balloon
    UNFAIR_STAMP = 1080                # Unfair Stamp (ACE SPEC)
    POFFIN = 1086                      # Buddy-Buddy Poffin
    HAMMER = 1120                      # Crushing Hammer
    DARK_ENERGY = 7                    # Basic {D} Energy

    SHADOW_BULLET = _nth_attack(GRIMMSNARL, 0)
    CORKSCREW_MORGREM = _nth_attack(MORGREM, 0)
    FILCH = _nth_attack(IMPIDIMP, 0)
    CORKSCREW_IMP = _nth_attack(IMPIDIMP, 1)
    MIND_BEND = _nth_attack(MUNKIDORI, 0)
    budew_atk = _nth_attack(BUDEW, 0)
    if budew_atk:
        ITCHY_POLLEN = budew_atk

    _ATTACK_BASE_DMG = {
        SHADOW_BULLET: 180, CORKSCREW_MORGREM: 60, CORKSCREW_IMP: 10,
        FILCH: 0, ITCHY_POLLEN: 10, MIND_BEND: 60,
    }
    _ATTACK_BASE_DMG.pop(0, None)


_resolve_ids()

# Board plan: 1 Froslass + 2 Grimmsnarl lines + 3 Munkidori = 6 slots exactly.
GRIMM_LINE_TARGET = 2
MUNKIDORI_TARGET = 3
FROSLASS_LINE_TARGET = 1

MARNIES_LINE = lambda: {IMPIDIMP, MORGREM, GRIMMSNARL}
FROSLASS_LINE = lambda: {SNORUNT, FROSLASS}
# Basics with 70 HP or less — the only things Buddy-Buddy Poffin can fetch.
POFFIN_TARGETS = lambda: {IMPIDIMP, SNORUNT, BUDEW}


# ── Log tracking (Item lock / knockout detection) ──

_cur_turn_logs = []
_item_locked = False
_we_got_koed = False


def _update_log_tracking(obs):
    global _item_locked, _we_got_koed, _cur_turn_logs
    yi = obs.current.yourIndex
    for entry in obs.logs:
        if entry.type == LogType.TURN_END:
            _item_locked = any(
                prev.type == LogType.ATTACK
                and getattr(prev, "playerIndex", yi) != yi
                and prev.attackId == ITCHY_POLLEN
                for prev in _cur_turn_logs)
            _we_got_koed = any(
                prev.type == LogType.MOVE_CARD
                and getattr(prev, "playerIndex", None) == yi
                and getattr(prev, "fromArea", None) in (AreaType.ACTIVE, AreaType.BENCH)
                and getattr(prev, "toArea", None) == AreaType.DISCARD
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


def all_opp_pokemon(obs):
    ps = opp_state(obs)
    return [p for p in (list(ps.active or []) + list(ps.bench or [])) if p]


def hand_ids(obs):
    return [c.id for c in (my_state(obs).hand or []) if c]


def energy_count(pokemon):
    if pokemon is None:
        return 0
    if getattr(pokemon, "energyCards", None) is not None:
        return len(pokemon.energyCards)
    return len(getattr(pokemon, "energies", []) or [])


def has_dark_energy(pokemon):
    """Adrena-Brain only switches on with a {D} Energy attached."""
    if pokemon is None:
        return False
    for c in (getattr(pokemon, "energyCards", None) or []):
        if getattr(c, "id", 0) == DARK_ENERGY:
            return True
        data = CARD_DB.get(getattr(c, "id", 0))
        # Special Energy that provides {D} also counts; be permissive.
        if data and getattr(data, "cardType", None) == CardType.SPECIAL_ENERGY:
            return True
    return False


def hp_of(card):
    """Remaining HP, or None if this isn't a Pokemon in play (a hand Card, etc.)."""
    return getattr(card, "hp", None)


def damage_on(pokemon):
    if pokemon is None:
        return 0
    hp = getattr(pokemon, "hp", None)
    if hp is None:
        return 0
    return max(0, getattr(pokemon, "maxHp", hp) - hp)


def count_in_play(obs, card_id):
    return sum(1 for p in all_my_pokemon(obs) if p.id == card_id)


def grimm_lines_in_play(obs):
    """Impidimp / Morgrem / Grimmsnarl all occupy one 'Grimmsnarl line' slot."""
    ids = MARNIES_LINE()
    return sum(1 for p in all_my_pokemon(obs) if p.id in ids)


def froslass_line_in_play(obs):
    ids = FROSLASS_LINE()
    return sum(1 for p in all_my_pokemon(obs) if p.id in ids)


def munkidori_in_play(obs):
    return count_in_play(obs, MUNKIDORI)


def armed_munkidori(obs):
    """Munkidori that already have {D} attached (Adrena-Brain is live)."""
    return sum(1 for p in all_my_pokemon(obs) if p.id == MUNKIDORI and has_dark_energy(p))


def bench_used(obs):
    return len([p for p in (my_state(obs).bench or []) if p])


def prize_value(pokemon):
    data = CARD_DB.get(pokemon.id) if pokemon else None
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def card_type(card_id):
    data = CARD_DB.get(card_id)
    return getattr(data, "cardType", None) if data else None


def has_ability(card_id):
    data = CARD_DB.get(card_id)
    if data is None:
        return False
    for attr in ("ability", "abilities", "abilityName"):
        v = getattr(data, attr, None)
        if v:
            return True
    return False


def is_dark_weak(pokemon):
    if pokemon is None:
        return False
    data = CARD_DB.get(pokemon.id)
    w = getattr(data, "weakness", None) if data else None
    if w is None:
        return False
    return getattr(w, "value", w) == DARK_ENERGY


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


def stadium_id(obs):
    for c in (obs.current.stadium or []):
        if c is not None:
            return c.id
    return 0


# ── Game plan helpers ──

def evolvable(obs, basic_id):
    """A Pokemon of this id in play that has been down since last turn."""
    return any(p.id == basic_id and not getattr(p, "appearThisTurn", True)
               for p in all_my_pokemon(obs))


def can_rare_candy(obs):
    """Impidimp that has sat a turn + Grimmsnarl ex in hand + Rare Candy in hand."""
    ids = hand_ids(obs)
    return (evolvable(obs, IMPIDIMP) and RARE_CANDY in ids
            and GRIMMSNARL in ids and not _item_locked)


def grimm_ready(pokemon):
    """A Grimmsnarl ex with enough Energy to fire Shadow Bullet."""
    return pokemon is not None and pokemon.id == GRIMMSNARL and energy_count(pokemon) >= 2


def bench_ready_grimm(obs):
    for i, p in enumerate(my_state(obs).bench or []):
        if grimm_ready(p):
            return i
    return -1


def expected_attack_damage(obs):
    """Rough best damage we can put out this turn (for Boss / KO planning)."""
    best = 0
    for p in all_my_pokemon(obs):
        e = energy_count(p)
        if p.id == GRIMMSNARL and e >= 2:
            best = max(best, 180)
        elif p.id == MORGREM and e >= 2:
            best = max(best, 60)
        elif p.id == IMPIDIMP and e >= 1:
            best = max(best, 10)
        elif p.id == BUDEW:
            best = max(best, 10)
    # Adrena-Brain can add up to 30 per armed Munkidori, if we have self-damage
    # to launder (Freezing Shroud reliably supplies it).
    movable = sum(min(3, damage_on(p) // 10) for p in all_my_pokemon(obs))
    best += 10 * min(movable, 3 * armed_munkidori(obs))
    return best


def marnies_energy_need(obs):
    """How much Punk Up fuel the Marnie's board still wants."""
    need = 0
    for p in all_my_pokemon(obs):
        if p.id == GRIMMSNARL:
            need += max(0, 2 - energy_count(p))
        elif p.id in (IMPIDIMP, MORGREM):
            need += max(0, 2 - energy_count(p))
    return need


def want_pokemon_id(obs):
    """The single Pokemon we most want to add to the board right now.
    Drives Spikemuth / Poke Pad / Poffin / Petrel targeting."""
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    # Grimmsnarl ex to evolve into is the whole deck's win condition.
    if GRIMMSNARL not in ids and count_in_play(obs, GRIMMSNARL) < GRIMM_LINE_TARGET \
            and (evolvable(obs, MORGREM) or can_rare_candy(obs)
                 or (evolvable(obs, IMPIDIMP) and RARE_CANDY in ids)) and dk[GRIMMSNARL] > 0:
        return GRIMMSNARL
    if grimm_lines_in_play(obs) < GRIMM_LINE_TARGET and dk[IMPIDIMP] > 0 and IMPIDIMP not in ids:
        return IMPIDIMP
    if munkidori_in_play(obs) < MUNKIDORI_TARGET and dk[MUNKIDORI] > 0 and MUNKIDORI not in ids:
        return MUNKIDORI
    if froslass_line_in_play(obs) < FROSLASS_LINE_TARGET and dk[SNORUNT] > 0 and SNORUNT not in ids:
        return SNORUNT
    if evolvable(obs, SNORUNT) and FROSLASS not in ids and count_in_play(obs, FROSLASS) < 1 \
            and dk[FROSLASS] > 0:
        return FROSLASS
    if GRIMMSNARL not in ids and count_in_play(obs, GRIMMSNARL) < GRIMM_LINE_TARGET and dk[GRIMMSNARL] > 0:
        return GRIMMSNARL
    if evolvable(obs, IMPIDIMP) and MORGREM not in ids and dk[MORGREM] > 0 \
            and count_in_play(obs, MORGREM) < 1:
        return MORGREM
    return 0


def board_is_set(obs):
    """Have we assembled (close to) the optimal board? Petrel pivots to
    disruption once we have."""
    return (grimm_lines_in_play(obs) >= GRIMM_LINE_TARGET
            and munkidori_in_play(obs) >= 2
            and count_in_play(obs, FROSLASS) >= 1)


def opp_has_energy(obs):
    return any(energy_count(p) > 0 for p in all_opp_pokemon(obs))


# ── Supporter planning ──

def desired_supporter(obs):
    """Pick the supporter we most want this turn (guideline priority).
    Returns (card_id, score)."""
    if obs.current.supporterPlayed:
        return 0, 0
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    my_hand = len(ids)
    opp_hand = getattr(opp_state(obs), "handCount", 0) or 0
    opp_bench = len(opp_bench_pokemon(obs))
    best = (0, 0)

    # Boss's Orders: drag up something we can Knock Out, or a fat prize target.
    our_dmg = expected_attack_damage(obs)
    for target in opp_bench_pokemon(obs):
        if target.hp <= our_dmg and prize_value(target) >= len(my_state(obs).prize):
            best = max(best, (BOSS, 60000), key=lambda x: x[1])  # lethal
        elif target.hp <= our_dmg and prize_value(target) >= 2:
            best = max(best, (BOSS, 42000 + prize_value(target) * 1000), key=lambda x: x[1])
        elif target.hp <= our_dmg and damage_on(target) > 0:
            best = max(best, (BOSS, 34000), key=lambda x: x[1])

    # Petrel: search out whatever Trainer builds the board (or disrupts).
    if PETREL in ids and petrel_wants(obs, dk):
        best = max(best, (PETREL, 33000), key=lambda x: x[1])

    # Judge: cut down a big opponent hand — but only when ours isn't the big one.
    if opp_hand >= 6 and my_hand <= 4:
        best = max(best, (JUDGE, 30000 + (opp_hand - my_hand) * 400), key=lambda x: x[1])

    # Morty's Conviction: our hand is already good, so don't shuffle it away —
    # just add cards off their Bench width (costs 1 discard).
    if my_hand >= 5 and opp_bench >= 3:
        best = max(best, (MORTY, 29500 + opp_bench * 900), key=lambda x: x[1])

    # Judge as a raw draw when we're truly empty and Lillie is absent.
    if my_hand <= 2 and LILLIE not in ids:
        best = max(best, (JUDGE, 24000), key=lambda x: x[1])

    # Lillie: the default refuel.
    need_setup = (grimm_lines_in_play(obs) < GRIMM_LINE_TARGET
                  or munkidori_in_play(obs) < 2
                  or count_in_play(obs, GRIMMSNARL) == 0)
    if need_setup or my_hand <= 4:
        best = max(best, (LILLIE, 26000 - my_hand * 800), key=lambda x: x[1])

    # Only offer supporters we actually hold.
    if best[0] and best[0] not in ids:
        return 0, 0
    return best


def petrel_wants(obs, dk):
    """Which Trainer would Petrel fetch? 0 if nothing is worth it.
    Guideline: build the board first (Poffin for Snorunt/Impidimp, or
    Spikemuth), then swap to Hammer disruption once we're set up."""
    ids = hand_ids(obs)
    # Spikemuth is the only Grimmsnarl ex searcher — get it down early.
    if stadium_id(obs) != SPIKEMUTH and SPIKEMUTH not in ids and dk[SPIKEMUTH] > 0:
        return SPIKEMUTH
    # Rare Candy line is the fastest clock in the deck.
    if evolvable(obs, IMPIDIMP) and GRIMMSNARL in ids and RARE_CANDY not in ids \
            and dk[RARE_CANDY] > 0 and not _item_locked:
        return RARE_CANDY
    # Bodies for the board.
    need_small = (grimm_lines_in_play(obs) < GRIMM_LINE_TARGET
                  or froslass_line_in_play(obs) < FROSLASS_LINE_TARGET)
    if need_small and POFFIN not in ids and dk[POFFIN] > 0 and not _item_locked:
        return POFFIN
    if want_pokemon_id(obs) and POKE_PAD not in ids and dk[POKE_PAD] > 0 and not _item_locked:
        return POKE_PAD
    # Board is together — turn Petrel into disruption.
    if board_is_set(obs) and opp_has_energy(obs) and dk[HAMMER] > 0 and not _item_locked:
        return HAMMER
    if dk[SPIKEMUTH] > 0 and stadium_id(obs) != SPIKEMUTH:
        return SPIKEMUTH
    if dk[HAMMER] > 0 and opp_has_energy(obs):
        return HAMMER
    return 0


# ── Setup ──

def score_setup(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ctx = obs.select.context

    if ctx == SelectContext.MULLIGAN:
        return (10000, "no mulligan") if opt.type == OptionType.NO else (0, "mulligan")
    if ctx == SelectContext.IS_FIRST:
        # Go second: extra card, and Budew can Itchy Pollen on turn 1 to lock
        # Items while this Stage 2 deck builds.
        return (10000, "choose second") if opt.type == OptionType.NO else (0, "go first")
    if ctx == SelectContext.SETUP_ACTIVE_POKEMON:
        # Budew leads: free retreat, 0-cost Item lock, and only 1 Prize.
        pri = {BUDEW: 100000, IMPIDIMP: 50000, SNORUNT: 20000,
               MUNKIDORI: 15000, MORGREM: 8000, FROSLASS: 5000, GRIMMSNARL: 1000}
        return pri.get(cid, 0), "setup Active"
    if ctx == SelectContext.SETUP_BENCH_POKEMON:
        # Free real estate before the game starts — fill toward the optimal board.
        if cid == IMPIDIMP and grimm_lines_in_play(obs) < GRIMM_LINE_TARGET:
            return 60000, "setup bench Impidimp"
        if cid == MUNKIDORI and munkidori_in_play(obs) < MUNKIDORI_TARGET:
            return 50000, "setup bench Munkidori"
        if cid == SNORUNT and froslass_line_in_play(obs) < FROSLASS_LINE_TARGET:
            return 40000, "setup bench Snorunt"
        if cid == BUDEW and count_in_play(obs, BUDEW) < 1:
            return 15000, "setup bench spare Budew"
        return -1000, "hold"
    return 0, "non-setup"


# ── MAIN-phase scoring ──

def score_play(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    free_bench = 5 - bench_used(obs)

    # ── Basics from hand ──
    if cid == IMPIDIMP:
        if free_bench <= 0:
            return -500, "bench full"
        if grimm_lines_in_play(obs) < GRIMM_LINE_TARGET:
            return 76000, "bench Impidimp (need 2 Grimmsnarl lines)"
        return -400, "Grimmsnarl lines full"
    if cid == MUNKIDORI:
        if free_bench <= 0:
            return -500, "bench full"
        if munkidori_in_play(obs) < MUNKIDORI_TARGET:
            return 75000, "bench Munkidori (want 3)"
        return -400, "Munkidori full"
    if cid == SNORUNT:
        if free_bench <= 0:
            return -500, "bench full"
        if froslass_line_in_play(obs) < FROSLASS_LINE_TARGET:
            return 74000, "bench Snorunt (want 1 Froslass)"
        if froslass_line_in_play(obs) < 2 and free_bench >= 2 and count_in_play(obs, FROSLASS) >= 1:
            return 8000, "bench backup Snorunt"
        return -400, "Froslass line full"
    if cid == BUDEW:
        if free_bench <= 0:
            return -500, "bench full"
        # Only an early-game disruption body; it costs a slot on the ideal board.
        if obs.current.turn <= 3 and count_in_play(obs, BUDEW) == 0 \
                and count_in_play(obs, GRIMMSNARL) == 0:
            return 30000, "bench Budew (early lock)"
        return -400, "no room for Budew"

    # ── Stadium ──
    if cid == SPIKEMUTH:
        if stadium_id(obs) == SPIKEMUTH:
            return -500, "Spikemuth already up"
        # Our only Grimmsnarl ex searcher, and it recurs every turn.
        return 95000, "play Spikemuth Gym"

    # ── Items ──
    if cid == RARE_CANDY:
        if can_rare_candy(obs):
            return 92000, "Rare Candy -> Grimmsnarl ex ASAP"
        return -500, "save Rare Candy"

    if cid == UNFAIR_STAMP:
        # ACE SPEC: only legal after we lost a Pokemon, so if it is offered
        # at all, take it. Guideline: always very good when playable.
        return 55000, "Unfair Stamp (always when legal)"

    if cid == HAMMER:
        if opp_has_energy(obs):
            return 65000, "Crushing Hammer whenever possible"
        return -500, "no Energy to hammer"

    if cid == POFFIN:
        # 70 HP or less only: Impidimp / Snorunt / Budew. Never Munkidori.
        if free_bench <= 0:
            return -500, "bench full"
        want_imp = grimm_lines_in_play(obs) < GRIMM_LINE_TARGET and dk[IMPIDIMP] > 0
        want_sno = froslass_line_in_play(obs) < FROSLASS_LINE_TARGET and dk[SNORUNT] > 0
        if want_imp or want_sno:
            return 70000, "Poffin: bench small basics"
        return -500, "nothing Poffin can fetch"

    if cid == POKE_PAD:
        # Any Pokemon WITHOUT a Rule Box — so Munkidori and Froslass, but
        # never Grimmsnarl ex.
        want = want_pokemon_id(obs)
        if want and want != GRIMMSNARL and dk[want] > 0:
            return 68000, "Poke Pad: fetch non-ex piece"
        if dk[MUNKIDORI] > 0 and munkidori_in_play(obs) < MUNKIDORI_TARGET and MUNKIDORI not in ids:
            return 66000, "Poke Pad: Munkidori (Poffin can't)"
        return -500, "save Poke Pad"

    if cid == AIR_BALLOON:
        return 60000, "attach Air Balloon"

    # ── Supporters (play only the planned one) ──
    if cid in (LILLIE, JUDGE, MORTY, PETREL, BOSS):
        want, _ = desired_supporter(obs)
        if cid != want:
            return -500, "not the planned supporter"
        if cid == PETREL:
            # Fetch early so the Trainer we grab is still playable this turn.
            return 72000, "Petrel: fetch a Trainer"
        if cid == BOSS:
            # Before attacking, after the board work.
            return 45000, "Boss's Orders: gust the target"
        if cid in (LILLIE, JUDGE):
            # These shuffle our hand away — do everything else first.
            return 18000, "refill hand (after items/attach)"
        return 19000, "Morty: draw off their Bench"

    if cid == GRIMMSNARL or cid == MORGREM or cid == FROSLASS:
        return -1000, "evolutions evolve, not played"
    return 1000, "generic play"


def score_evolve(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None
    is_active = getattr(opt, "inPlayArea", None) == AreaType.ACTIVE

    if cid == GRIMMSNARL:
        # Punk Up fires on evolve from hand — this is our energy engine.
        s = 91000 + energy_count(target) * 10
        if is_active:
            s += 4000  # start attacking this turn
        if count_in_play(obs, GRIMMSNARL) >= GRIMM_LINE_TARGET:
            s = 6000
        return s, "evolve Grimmsnarl ex (Punk Up)"
    if cid == FROSLASS:
        # Freezing Shroud is the damage engine feeding Adrena-Brain.
        if count_in_play(obs, FROSLASS) >= 1:
            # A 2nd Froslass puts 20/turn on each of our own Ability Pokemon
            # (~100 across the board) — more than 3 Munkidori can launder off.
            return -500, "one Froslass is the plan"
        return 89000, "evolve Froslass (Freezing Shroud)"
    if cid == MORGREM:
        if count_in_play(obs, GRIMMSNARL) >= GRIMM_LINE_TARGET:
            return 4000, "lines already done"
        return 80000, "evolve Morgrem (toward Grimmsnarl)"
    return 10000 + energy_count(target), "generic evolve"


def score_ability(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else None

    if cid == SPIKEMUTH:
        # Once per turn, free: search out a Marnie's Pokemon. Always take it.
        if my_state(obs).deckCount <= 1:
            return -500, "deck too thin"
        return 98000, "Spikemuth Gym: search a Marnie's Pokemon"

    if cid == MUNKIDORI:
        # Adrena-Brain: needs {D} attached and damage on one of our Pokemon.
        if not has_dark_energy(card):
            return -500, "Munkidori has no {D}"
        if not any(damage_on(p) >= 10 for p in all_my_pokemon(obs)):
            return -500, "no self-damage to move yet"
        if not all_opp_pokemon(obs):
            return -500, "no target"
        # After Energy attach (22000) so a freshly armed Munkidori can fire too.
        return 20000, "Adrena-Brain: launder damage onto them"

    return 1, "generic ability"


def attach_target_score(obs, target, in_active, attach_id):
    """Score attaching Energy onto `target`.

    Core guideline: Grimmsnarl powers itself with Punk Up (Marnie's Pokemon
    only), so the hand attachment belongs on Munkidori, which Punk Up can
    never reach and which needs exactly one {D} to switch Adrena-Brain on."""
    if target is None:
        return -500
    tid = target.id
    e = energy_count(target)

    if tid == MUNKIDORI:
        if has_dark_energy(target):
            return -600  # one is all Adrena-Brain ever needs
        score = 30000
        if munkidori_in_play(obs) >= 2:
            score += 500  # more Munkidori = more counters moved per turn
        return score

    if tid == GRIMMSNARL:
        # Only hand-feed Grimmsnarl if Punk Up hasn't covered it and we need
        # to attack now.
        if e >= 2:
            return -600
        if armed_munkidori(obs) < min(MUNKIDORI_TARGET, munkidori_in_play(obs)):
            return 12000 if in_active else 8000  # Munkidori still comes first
        return 24000 if in_active else 18000

    if tid in (IMPIDIMP, MORGREM):
        # Pre-loading the line is fine, but it is the lowest priority — Punk Up
        # can top these up for free later.
        if e >= 2:
            return -600
        if armed_munkidori(obs) < munkidori_in_play(obs):
            return 3000
        return 9000 if count_in_play(obs, GRIMMSNARL) < GRIMM_LINE_TARGET else 2000

    if tid in (BUDEW, SNORUNT, FROSLASS):
        return -800  # none of these ever attack for us

    return 1000


def score_attach(obs, opt):
    card = option_card(obs, opt)
    target = option_target(obs, opt)
    cid = card.id if card else None

    tool_type = getattr(CardType, "TOOL", None)
    if cid == AIR_BALLOON or (tool_type is not None and card_type(cid) == tool_type):
        # Grimmsnarl retreats for 2 — the balloon makes the pivot free.
        if target is None:
            return 60000, "attach tool"
        if target.id == GRIMMSNARL:
            return 62000, "Air Balloon on Grimmsnarl"
        if target.id in (BUDEW, SNORUNT):
            return -500, "wasted balloon"
        return 40000, "attach tool"

    if obs.current.energyAttached:
        return -1000, "already attached this turn"
    in_active = getattr(opt, "inPlayArea", None) == AreaType.ACTIVE
    s = attach_target_score(obs, target, in_active, cid)
    # Keep negatives negative so they stay skippable; only lift real targets.
    return (s + 1000 if s > 0 else s), "attach Energy"


def score_attach_from(obs, opt):
    """Punk Up: distributing up to 5 Basic {D} from the deck.
    Marnie's Pokemon ONLY — Munkidori is not a legal target here."""
    target = option_target(obs, opt) or option_card(obs, opt)
    if target is None:
        return 1000, "punk up"
    tid = getattr(target, "id", 0)
    e = energy_count(target)
    in_active = getattr(opt, "inPlayArea", None) == AreaType.ACTIVE \
        or getattr(opt, "area", None) == AreaType.ACTIVE

    if tid == GRIMMSNARL:
        if e >= 2:
            return 2000, "Grimmsnarl already armed"
        return (40000 if in_active else 34000) - e * 100, "Punk Up: arm Grimmsnarl"
    if tid == MORGREM:
        if e >= 2:
            return 1500, "Morgrem loaded"
        return 20000, "Punk Up: pre-load Morgrem"
    if tid == IMPIDIMP:
        if e >= 2:
            return 1200, "Impidimp loaded"
        return 15000, "Punk Up: pre-load Impidimp"
    return 500, "Punk Up: other"


def score_retreat(obs, opt):
    act = active_pokemon(obs)
    if act is None:
        return -100, "no active"
    ready = bench_ready_grimm(obs)
    # The whole plan: disrupt with Budew, then pivot into Grimmsnarl and swing.
    if ready >= 0 and act.id != GRIMMSNARL:
        return 12000, "pivot into a ready Grimmsnarl"
    if act.id in (SNORUNT, FROSLASS, MUNKIDORI) and ready >= 0:
        return 12500, "get the support Pokemon out of the Active spot"
    if act.id == BUDEW and count_in_play(obs, GRIMMSNARL) >= 1 and ready >= 0:
        return 12000, "Budew has done its job"
    return -100, "hold position"


def score_attack(obs, opt):
    aid = getattr(opt, "attackId", None)
    act = active_pokemon(obs)
    opp = opp_active_pokemon(obs)
    base = _ATTACK_BASE_DMG.get(aid, 0)
    if base == 0 and aid in ALL_ATTACKS:
        base = getattr(ALL_ATTACKS[aid], "damage", 0) or 0

    dmg = base
    if opp is not None and is_dark_weak(opp) and act is not None and act.id in MARNIES_LINE():
        dmg = base * 2

    score = 1000 + dmg
    if opp is not None and hp_of(opp) is not None and dmg >= opp.hp:
        score += 5000 + prize_value(opp) * 1000  # take the KO

    if aid == SHADOW_BULLET:
        score += 600  # the 30 bench chip is real value every single turn
    elif aid == ITCHY_POLLEN:
        # Item lock is worth far more than 10 damage while we are still
        # assembling the board — but never stall once Grimmsnarl can swing.
        if count_in_play(obs, GRIMMSNARL) == 0:
            score += 4000
        else:
            score -= 2000
    elif aid == FILCH:
        # Only ever a turn-1 filler when nothing else can happen.
        score += 200 if obs.current.turn <= 2 else -500
    return score, "attack"


# ── Selection scoring (searches / discards / targets) ──

def score_to_hand(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    effect = getattr(obs.select, "effect", None)
    effect_id = effect.id if effect else None
    want = want_pokemon_id(obs)

    # Spikemuth Gym: Marnie's Pokemon only.
    if effect_id == SPIKEMUTH:
        if cid == GRIMMSNARL:
            # Grimmsnarl ex has a Rule Box, so this Stadium is the only way to
            # dig it out of the deck.
            if count_in_play(obs, GRIMMSNARL) < GRIMM_LINE_TARGET and GRIMMSNARL not in ids:
                return 40000, "Spikemuth: Grimmsnarl ex"
            return 12000, "Spikemuth: spare Grimmsnarl"
        if cid == MORGREM and evolvable(obs, IMPIDIMP) and MORGREM not in ids \
                and count_in_play(obs, MORGREM) < 1:
            return 25000, "Spikemuth: Morgrem"
        if cid == IMPIDIMP and grimm_lines_in_play(obs) < GRIMM_LINE_TARGET:
            return 22000, "Spikemuth: Impidimp"
        return 3000, "Spikemuth: generic Marnie's"

    # Team Rocket's Petrel: any Trainer.
    if effect_id == PETREL:
        target = petrel_wants(obs, dk)
        if cid == target:
            return 40000, "Petrel: planned Trainer"
        pri = {SPIKEMUTH: 26000, RARE_CANDY: 24000, POFFIN: 22000,
               POKE_PAD: 20000, HAMMER: 18000, BOSS: 16000,
               UNFAIR_STAMP: 15000, LILLIE: 14000, AIR_BALLOON: 6000}
        return pri.get(cid, 5000), "Petrel: fallback Trainer"

    # Poke Pad: Pokemon without a Rule Box.
    if effect_id == POKE_PAD:
        if cid == want and cid != GRIMMSNARL:
            return 40000, "Pad: the piece we need"
        if cid == MUNKIDORI and munkidori_in_play(obs) < MUNKIDORI_TARGET:
            return 30000, "Pad: Munkidori"
        if cid == FROSLASS and evolvable(obs, SNORUNT) and count_in_play(obs, FROSLASS) < 1:
            return 28000, "Pad: Froslass"
        if cid == IMPIDIMP and grimm_lines_in_play(obs) < GRIMM_LINE_TARGET:
            return 26000, "Pad: Impidimp"
        if cid == SNORUNT and froslass_line_in_play(obs) < FROSLASS_LINE_TARGET:
            return 24000, "Pad: Snorunt"
        if cid == MORGREM and evolvable(obs, IMPIDIMP):
            return 20000, "Pad: Morgrem"
        return 2000, "Pad: generic"

    # Punk Up digging Basic {D} out of the deck.
    if effect_id == GRIMMSNARL:
        if cid == DARK_ENERGY:
            return 40000, "Punk Up: take the Energy"
        return -500, "Punk Up: Energy only"

    # Generic to-hand ranking.
    if cid == want:
        return 30000, "take the piece we need"
    if cid == GRIMMSNARL and count_in_play(obs, GRIMMSNARL) < GRIMM_LINE_TARGET:
        return 28000, "take Grimmsnarl ex"
    if cid == RARE_CANDY and evolvable(obs, IMPIDIMP):
        return 26000, "take Rare Candy"
    if cid == MUNKIDORI and munkidori_in_play(obs) < MUNKIDORI_TARGET:
        return 22000, "take Munkidori"
    if cid == SPIKEMUTH and stadium_id(obs) != SPIKEMUTH:
        return 20000, "take Spikemuth"
    if cid == DARK_ENERGY and armed_munkidori(obs) < munkidori_in_play(obs):
        return 18000, "take {D} for Munkidori"
    if cid in (LILLIE, JUDGE, MORTY, PETREL, BOSS):
        return 8000, "take supporter"
    return 1000, "generic take"


def score_to_bench(obs, opt):
    """Poffin / Poke Pad putting Basics straight onto the Bench."""
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    if bench_used(obs) >= 5:
        return -500, "bench full"
    if cid == IMPIDIMP and grimm_lines_in_play(obs) < GRIMM_LINE_TARGET:
        return 30000, "bench Impidimp"
    if cid == MUNKIDORI and munkidori_in_play(obs) < MUNKIDORI_TARGET:
        return 28000, "bench Munkidori"
    if cid == SNORUNT and froslass_line_in_play(obs) < FROSLASS_LINE_TARGET:
        return 26000, "bench Snorunt"
    if cid == BUDEW and count_in_play(obs, BUDEW) == 0 and obs.current.turn <= 3:
        return 10000, "bench Budew"
    return -500, "board is where we want it"


# Discard value: LOWER value = happier to discard.
def _keep_value(obs, cid):
    ids = hand_ids(obs)
    dk = deck_counts(obs)
    if cid == GRIMMSNARL:
        return 9500
    if cid == RARE_CANDY and evolvable(obs, IMPIDIMP):
        return 9000
    if cid == UNFAIR_STAMP:
        return 9800 if _we_got_koed else 8500
    if cid == BOSS:
        return 8000
    if cid == SPIKEMUTH:
        return 500 if ids.count(SPIKEMUTH) > 1 or stadium_id(obs) == SPIKEMUTH else 7000
    if cid == DARK_ENERGY:
        # Punk Up pulls these from the DECK, so discarding them thins our own
        # acceleration. Keep them while the count is low.
        return 6500 if dk[DARK_ENERGY] <= 4 else 3000
    if cid == MUNKIDORI:
        return 6000 if munkidori_in_play(obs) < MUNKIDORI_TARGET else 2500
    if cid == IMPIDIMP:
        return 5500 if grimm_lines_in_play(obs) < GRIMM_LINE_TARGET else 2000
    if cid == FROSLASS or cid == SNORUNT:
        return 5000 if count_in_play(obs, FROSLASS) < 1 else 1800
    if cid == PETREL:
        return 4500
    if cid == MORGREM:
        return 4000
    if cid == LILLIE:
        return 3000
    if cid in (JUDGE, MORTY):
        return 2600
    if cid in (POKE_PAD, POFFIN, HAMMER):
        return 2200
    if cid == BUDEW:
        return 2500 if count_in_play(obs, BUDEW) == 0 and obs.current.turn <= 3 else 900
    if cid == AIR_BALLOON:
        return 1200
    return 2000


def score_discard(obs, opt):
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ids = hand_ids(obs)
    score = 10000 - _keep_value(obs, cid)
    if ids.count(cid) > 1:
        score += 4000  # duplicates go first
    return score, "discard"


def score_opp_energy(obs, opt):
    """Crushing Hammer: Special Energy first, then the Active."""
    card = None
    try:
        card = get_card(obs, opt.area, opt.index, opt.playerIndex)
    except Exception:
        pass
    on_active = getattr(opt, "area", None) == AreaType.ACTIVE
    is_special = bool(card and card_type(card.id) == CardType.SPECIAL_ENERGY)
    score = 1000
    score += 3000 if is_special else 0   # guideline: prioritize Special Energy
    score += 1500 if on_active else 0
    return score, "hammer target"


def score_damage_counter(obs, opt):
    """Adrena-Brain source (ours) and destination (theirs) selection."""
    card = option_card(obs, opt)
    if card is None:
        return 0, "no target"
    yi = obs.current.yourIndex
    pi = getattr(opt, "playerIndex", yi)

    hp = hp_of(card)
    if hp is None:
        return 0, "not a Pokemon in play"

    if pi == yi:
        # Where do we pull counters FROM? Keep the attacker healthy first.
        dmg = damage_on(card)
        if dmg <= 0:
            return -500, "undamaged"
        s = 1000 + dmg
        if card.id == GRIMMSNARL:
            s += 8000   # our 2-Prize attacker — unload it
        elif card.id == MUNKIDORI:
            s += 3000   # it self-heals the board every turn anyway
        elif card.id in (IMPIDIMP, MORGREM, SNORUNT):
            s += 500
        if hp <= 30:
            s += 4000   # about to fall over
        return s, "Adrena-Brain source"

    # Where do the counters GO? Finish something off.
    s = 1000 + prize_value(card) * 800 + energy_count(card) * 100
    remain = (getattr(obs.select, "remainDamageCounter", 3) or 3) * 10
    if 0 < hp <= remain:
        s += 20000 + prize_value(card) * 2000   # Knock Out right now
    elif damage_on(card) > 0:
        s += 3000                               # keep stacking a wounded target
    if has_ability(card.id):
        s += 1500   # Freezing Shroud keeps chipping these for us
    s -= hp // 10
    return s, "Adrena-Brain target"


def score_target(obs, opt):
    """SWITCH / TO_ACTIVE / HEAL / DAMAGE etc."""
    card = option_card(obs, opt)
    cid = card.id if card else getattr(opt, "cardId", None)
    ctx = obs.select.context
    yi = obs.current.yourIndex
    pi = getattr(opt, "playerIndex", yi)

    if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
               SelectContext.DAMAGE_COUNTER_ANY):
        return score_damage_counter(obs, opt)

    if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE,
               SelectContext.SETUP_ACTIVE_POKEMON):
        if pi != yi and card is not None and hp_of(card) is not None:
            # Boss's Orders: drag up whatever we can Knock Out.
            dmg = expected_attack_damage(obs)
            s = prize_value(card) * 1000 + energy_count(card) * 100
            if card.hp <= dmg:
                s += 20000 + damage_on(card)
            elif damage_on(card) > 0:
                s += 3000
            return s, "Boss target"
        # Our promotion order: a loaded Grimmsnarl, else Budew to stall.
        e = energy_count(card) if card else 0
        pri = {GRIMMSNARL: 20000, BUDEW: 12000, MORGREM: 9000,
               IMPIDIMP: 8000, MUNKIDORI: 4000, FROSLASS: 2000, SNORUNT: 1500}
        s = pri.get(cid, 1000)
        if cid == GRIMMSNARL and e >= 2:
            s += 6000
        elif cid in (MORGREM, IMPIDIMP):
            s += e * 500
        return s, "promote"

    if ctx == SelectContext.HEAL:
        if cid == GRIMMSNARL:
            return 20000 + damage_on(card), "heal Grimmsnarl"
        return damage_on(card), "heal"

    if ctx == SelectContext.ATTACH_FROM:
        return score_attach_from(obs, opt)

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
        # "Move up to 3 damage counters", "draw up to N" — always take the max.
        return (getattr(opt, "number", 0) or 0), "number (take max)"

    if opt.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        yi = obs.current.yourIndex
        if getattr(opt, "playerIndex", yi) != yi:
            return score_opp_energy(obs, opt)
        # Paying our own cost (retreat etc.): shed from a non-attacker first.
        if getattr(opt, "area", None) == AreaType.ACTIVE and active_pokemon(obs) \
                and active_pokemon(obs).id == GRIMMSNARL:
            return 500, "avoid stripping Grimmsnarl"
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
    if ctx == SelectContext.ATTACH_FROM:
        return score_attach_from(obs, opt)
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

    # Forced hand discards: greedy pick with card-TYPE diversity so we don't
    # strip every copy of one resource.
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
                    s2 -= 3000 * picked_types[ct]
                if cid is not None and picked_ids[cid] > 0:
                    s2 += 1500
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
        global _cur_turn_logs, _item_locked, _we_got_koed
        _cur_turn_logs.clear()
        _item_locked = False
        _we_got_koed = False
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
