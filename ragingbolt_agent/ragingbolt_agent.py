import os
import sys
import random
from collections import defaultdict
import json
import pandas as pd

from ptcg.api import (
  AreaType,
  CardType,
  Log,
  LogType,
  Observation,
  SelectContext,
  SelectType,
  OptionType,
  Card,
  Pokemon,
  State,
  all_card_data,
  to_observation_class
)

TYPE_PLAY = 7
TYPE_ATTACH = 8
TYPE_ABILITY = 10
TYPE_ATTACK = 13
TYPE_END = 14

MAIN = 0

DEBUG = True

CARD_NAME = {
    1: "Basic {G} Energy",
    4: "Basic {L} Energy",
    5: "Basic {P} Energy",
    6: "Basic {F} Energy",
    63: "Raging Bolt ex",
    96: "Teal Mask Ogerpon ex",
    140: "Fezandipiti ex",
    176: "Terapagos ex",
    184: "Latias ex",
    756: "Mega Kangaskhan ex",
    1094: "Bug Catching Set",
    1098: "Glass Trumpet",
    1102: "Dusk Ball",
    1116: "Energy Switch",
    1118: "Energy Retrieval",
    1121: "Ultra Ball",
    1125: "Master Ball",
    1182: "Boss’s Orders",
    1198: "Crispin",
    1205: "Cyrano",
    1210: "Brock’s Scouting",
    1227: "Lillie's Determination",
    1250: "Area Zero Underdepths",
}

def card_name_from_game_id(card_id):
    return CARD_NAME.get(int(card_id), f"UNKNOWN_{card_id}")

def read_deck_csv() -> list[int]:
# Load deck.csv: package-local first, then the Kaggle agent directory.
import os as _os

_deck_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "deck.csv")
if not _os.path.exists(_deck_path):
    _deck_path = "/kaggle_simulations/agent/deck.csv"
with open(_deck_path) as _f:
    my_deck: list[int] = [int(_line) for _line in _f.read().splitlines() if _line.strip()]


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py."""
    global my_deck
    my_deck = list(deck_list)

def get_my_player(obs):
    return obs.current.players[obs.current.yourIndex]

def card_value_by_name(name):
    if name == "Basic {G} Energy":
        return 25
    if name == "Basic {L} Energy":
        return 20
    if name == "Basic {F} Energy":
        return 15
    if name == "Basic {P} Energy":
        return 10

    if name == "Boss’s Orders":
        return 20

    if name in {
        "Raging Bolt ex",
        "Teal Mask Ogerpon ex",
        "Fezandipiti ex",
        "Mega Kangaskhan ex",
    }:
        return 80

    if name == "Latias ex":
        return 70

    if name == "Terapagos ex":
        return 65

    if name in {
        "Crispin",
        "Lillie's Determination",
        "Boss's Orders",
        "Ultra Ball",
    }:
        return 50
    
    if name == "Energy Switch":
        return 5

    return 30

def play_priority(obs, name):
    if name == "Teal Mask Ogerpon ex":
        return 100

    if name == "Raging Bolt ex":
        return 95
    
    if name == "Fezandipiti ex":
        return 80
    
    if name == "Latias ex":
        return 70
    
    if name == "Terapagos ex":
        return 65
    
    if name ==  "Mega Kangaskhan ex":
        return 60
    
    if name == "Lillie's Determination":
        return 90
    
    if name == "Brock’s Scouting":
        return 85

    if name == "Bug Catching Set":
        return 80

    if name == "Crispin":
        if count_energy_in_hand(obs) >= 2:
            return 20
        return 75
    
    if name == "Master Ball":
        return 70

    if name == "Dusk Ball":
        return 65
    
    if name == "Area Zero Underdepths":
        return 50
    
    if name == "Energy Retrieval":
        return 25
    
    if name == "Energy Switch":
        return 0
    
    if name == "Glass Trumpet":
        return 60

    if name == "Boss’s Orders":
        return 5

    return 40

ENERGY_NAMES = {
    "Basic {G} Energy",
    "Basic {L} Energy",
    "Basic {F} Energy",
    "Basic {P} Energy",
}

MAIN_ATTACKERS = [
    "Raging Bolt ex",
    "Teal Mask Ogerpon ex",
    "Mega Kangaskhan ex",
]

BENCH_PRIORITY = [
    "Raging Bolt ex",
    "Teal Mask Ogerpon ex",
    "Mega Kangaskhan ex",
    "Latias ex",
    "Fezandipiti ex",
    "Terapagos ex",
]

ABILITY_PRIORITY = [
    "Teal Mask Ogerpon ex",
    "Fezandipiti ex",
    "Latias ex",
]


def option_hand_card_name(obs, op):
    my = obs.current.players[obs.current.yourIndex]
    if op.index is None or my.hand is None:
        return None
    return card_name_from_game_id(my.hand[op.index].id)

def get_target_pokemon(obs, op):
    my = obs.current.players[obs.current.yourIndex]

    if op.inPlayArea == AreaType.ACTIVE:
        return my.active[op.inPlayIndex]
    if op.inPlayArea == AreaType.BENCH:
        return my.bench[op.inPlayIndex]
    return None



def count_energy_in_hand(obs):
    my = obs.current.players[obs.current.yourIndex]
    hand = my.hand or []

    cnt = 0

    for card in hand:
        name = card_name_from_game_id(card.id)
        if name in ENERGY_NAMES:
            cnt += 1

    return cnt

def get_attach_target(option):
    if option.inPlayArea is None or option.inPlayIndex is None:
        return None
    return option.inPlayArea, option.inPlayIndex

def target_is_active(option):
    target = get_attach_target(option)
    if target is None:
        return False

    area, index = target
    return area == 0


def score_basic_to_bench(card):
    name = card.name
    for i, key in enumerate(BENCH_PRIORITY):
        if key in name:
            return 100 - i * 10

    if "ex" in name:
        return 30

    return 0

def ability_score(obs, op):
    target = get_target_pokemon(obs, op)
    if target is None:
        return 0

    name = card_name_from_game_id(target.id)

    score = 0

    if name == "Teal Mask Ogerpon ex":
        score += 100
    elif name == "Fezandipiti ex":
        score += 80
    elif name == "Latias ex":
        score += 40
    else:
        score += 10

    if op.inPlayArea == AreaType.ACTIVE:
        score += 5

    return score

def choose_first(options, option_type):
    for i, op in enumerate(options):
        if op.type == option_type:
            return [i]
    return None

def choose_attach(options):
    for i, op in enumerate(options):
        if op.type == TYPE_ATTACH and op.inPlayArea == 4:
            return [i]
    for i, op in enumerate(options):
        if op.type == TYPE_ATTACH and op.inPlayArea == 5:
            return [i]
    return None

def should_play_ultra_ball(obs):
    my = obs.current.players[obs.current.yourIndex]
    hand = my.hand or []

    if len(hand) < 7:
        return False

    low = 0

    for card in hand:
        name = card_name_from_game_id(card.id)
        if card_value_by_name(name) <= 30:
            low += 1

    return low >= 2

    energy_cnt = count_energy_in_hand(obs)
    if energy_cnt <= 2:
        return False

    return True

def choose_discard_low_value(obs):
    hand = obs.current.players[obs.current.yourIndex].hand
    values = []

    for option_i, op in enumerate(obs.select.option):
        name = card_name_from_game_id(hand[op.index].id)
        v = card_value_by_name(name)
        values.append((v, option_i, op.index, name))

    values.sort()
    print("DISCARD_CANDIDATES", values)

    return [values[0][1], values[1][1]]

def choose_good_play(obs, options):
    best_i = None
    best_score = -10**9

    for i, op in enumerate(options):
        if op.type != OptionType.PLAY:
            continue

        name = option_hand_card_name(obs, op)

        if name == "Ultra Ball" and not should_play_ultra_ball(obs):
            continue

        if name == "Area Zero Underdepths" and obs.current.stadium:
            continue

        if name == "Energy Retrieval" and count_energy_in_hand(obs) >= 2:
            continue

        if name == "Crispin" and count_energy_in_hand(obs) >= 3:
            continue

        if name == "Energy Switch":
            continue

        score = play_priority(obs, name)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None
    return [best_i]

def choose_effect_choice(obs):
    context = obs.select.context
    options = obs.select.option
    
    if context == SelectContext.IS_FIRST:
        return [0]
    
    if context == SelectContext.MULLIGAN:
        return [0]
    
    if context in [
        SelectContext.SETUP_ACTIVE_POKEMON,
        SelectContext.SETUP_BENCH_POKEMON,
        SelectContext.TO_ACTIVE,
        SelectContext.TO_BENCH,
        SelectContext.TO_FIELD,
    ]:
        return [0]

    if context in [
        SelectContext.TO_HAND,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.NOT_MOVE,
        ]:
        return choose_required_count(obs.select)
        
    if context in [
        SelectContext.DISCARD,
        ]:
        return choose_discard_low_value(obs)


    if context in [
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
    ]:
        return [0]
    
    if context in [
        SelectContext.ATTACH_FROM,
        SelectContext.ATTACH_TO,
        SelectContext.DETACH_FROM,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_TOOL_CARD,
        SelectContext.SWITCH_ENERGY_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.TO_HAND_ENERGY,
        SelectContext.TO_DECK_ENERGY,
        SelectContext.SWITCH_ENERGY,
    ]:
        return [0]
    
    if context == SelectContext.ATTACK:
        return [len(options) - 1]

    if context in [
        SelectContext.ACTIVATE,
        SelectContext.FIRST_EFFECT,
        SelectContext.MORE_DEVOLVE,
        SelectContext.COIN_HEAD,
    ]:
        return [0]

    return [0]

def get_select_count(select):
    for attr in ["count", "number", "min", "max"]:
        if hasattr(select, attr):
            v = getattr(select, attr)
            if v is not None:
                print("select", attr, v)

def choose_required_count(select):
    n = select.maxCount
    return list(range(min(n, len(select.option))))


def safe_first_action(obs):
  n = len(obs.select.option)
  min_c = obs.select.minCount
  max_c = obs.select.maxCount

  if max_c == 0:
    return []

  if n == 0:
    raise RuntimeError("No options, but selection requested")
  
  k = min_c
  
  if k == 0 and max_c > 0:
    k = 1
  
  k = min(k, max_c, n)
  
  return list(range(k))

def log(*args):
    with open("agent_summary.log", "a", encoding="utf-8") as f:
        f.write(" ".join(map(str, args)))
        f.write("\n")

def enum_name(enum_cls, value):
    for name in dir(enum_cls):
        if name.startswith("_"):
            continue
        try:
            if getattr(enum_cls, name) == value:
                return name
        except Exception:
            pass
    return str(value)



def summarize_obs(obs):
    s = obs.select
    log("=" * 80)
    log("context =", enum_name(SelectContext, s.context))
    log("min/max =", s.minCount, s.maxCount)
    log("n_options =", len(s.option))
    log("obs dir =", dir(obs))
    log("obs repr =", repr(obs)[:3000])


    for i, op in enumerate(s.option):
        log(f"[{i}] type={enum_name(OptionType, op.type)}")
        log(
            "   ",
            "area=", enum_name(AreaType, op.area),
            "index=", op.index,
            "player=", op.playerIndex,
            "cardId=", op.cardId,
            "attackId=", op.attackId,
        )


def debug_hand(state):
    me = state.players[state.yourIndex]

    print("=== HAND ===")

    for i, card in enumerate(me.hand):
        print(i, card)

def get_card_from_option(obs, op):
    if op.cardId is not None:
        return op.cardId

def debug_options(obs, options):
    if not DEBUG:
        return

    my_player = obs.current.players[obs.current.yourIndex]

    print("===== HAND =====")
    if my_player.hand is not None:
        for i, card in enumerate(my_player.hand):
            print(i, card_name_from_game_id(card.id))

    print("===== OPTIONS =====")
    for i, op in enumerate(options):
        print(i, op.type, op.index)


def attach_score(obs, op):
    energy_name = option_hand_card_name(obs, op)
    if energy_name is None:
        return -10**9
    target = get_target_pokemon(obs, op)
    if target is None:
        return -10**9

    target_name = card_name_from_game_id(target.id)
    score = 0

    if energy_name == "Basic {G} Energy":
        if target_name == "Teal Mask Ogerpon ex":
            score += 100
        elif target_name == "Raging Bolt ex":
            score += 60

    elif energy_name == "Basic {L} Energy":
        if target_name == "Raging Bolt ex":
            score += 100
        elif target_name == "Teal Mask Ogerpon ex":
            score += 40

    elif energy_name == "Basic {F} Energy":
        if target_name == "Mega Kangaskhan ex":
            score += 100
        elif target_name == "Raging Bolt ex":
            score += 40

    hp = getattr(target, "hp", 999)
    if hp <= 50:
        score -= 80

    if op.inPlayArea == AreaType.ACTIVE:
        score += 20

    return score

def choose_best_attach(obs, options):
    my_player = get_my_player(obs)

    best_i = None
    best_score = -1

    for i, op in enumerate(options):
        if op.type != OptionType.ATTACH:
            continue

        if op.index is None:
            continue

        if op.index >= len(my_player.hand):
            continue

        card = my_player.hand[op.index]
        name = card_name_from_game_id(card.id)

        score = attach_score(obs, op)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score > 0:
        return [best_i]

    return None

def choose_best_attach_energy(obs, options):
    attach_options = [
        (i, op) for i, op in enumerate(options)
        if op.type == OptionType.ATTACH
    ]

    if not attach_options:
        return None

    return [attach_options[0][0]]

def choose_attack(options):
    attacks = [i for i, op in enumerate(options) if op.type == OptionType.ATTACK]
    if attacks:
        return [attacks[-1]]
    return None

def score_bench_play(option, obs):
    name = option_hand_card_name(obs, option)

    score = 0
    for i, key in enumerate(BENCH_PRIORITY):
        if key in name:
            score += 100 - i * 10

    if "ex" in name:
        score += 20

    return score

def score_attach(option, obs):
    target = get_attach_target(option)

    score = 0

    if target_is_active(target):
        score += 100

    name = get_target_pokemon(target, obs)

    for key in MAIN_ATTACKERS:
        if key in name:
            score += 50

    energy_count = count_energy_in_hand(target, obs)
    score += max(0, 30 - energy_count * 10)

    return score


def score_basic_to_bench(name):
    for i, target in enumerate(BENCH_PRIORITY):
        if name == target:
            return 100 - i * 10

    return 0

def choose_best_play_basic_to_bench(obs, options):
    my_player = get_my_player(obs)

    best_i = None
    best_score = -1

    for i, op in enumerate(options):
        if op.type != OptionType.PLAY:
            continue

        if op.index is None:
            continue

        if op.index >= len(my_player.hand):
            continue

        card = my_player.hand[op.index]
        name = card_name_from_game_id(card.id)

        score = score_basic_to_bench(name)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None and best_score > 0:
        return [best_i]

    return None

def choose_best_ability(obs, options):
    best_i = None
    best_score = -10**9

    for i, op in enumerate(options):
        if op.type != OptionType.ABILITY:
            continue

        score = ability_score(obs, op)

        if score > best_score:
            best_score = score
            best_i = i

    if best_i is not None:
        return [best_i]

    return None

def agent(obs_dict: dict) -> list[int]:
    obs: Observation = to_observation_class(obs_dict)

    if obs.select is None:
        return read_deck_csv()

    options = obs.select.option

    debug_options(obs, options)
    
    if obs.select.context == SelectContext.MAIN:
        res = choose_best_play_basic_to_bench(obs, options)
        if res is not None:
            return res
        

        res = choose_best_attach(obs, options)
        if res is not None:
            return res

        res = choose_good_play(obs, options)
        if res is not None:
            return res

        res = choose_best_ability(obs, options)
        if res is not None:
            return res

        res = choose_attack(options)
        if res is not None:
            return res

        res = choose_first(options, OptionType.END)
        if res is not None:
            return res
    
    else:
        return choose_effect_choice(obs)

    return safe_first_action(obs)
