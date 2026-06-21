import os
import random

deck_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
with open(deck_path) as _f:
    deck = [int(line) for line in _f.readlines() if line.strip()]


def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")

    # First turn, no choice selection is requested, so must return deck
    if select is None:
        return deck

    options = select.get("option", [])
    if not options:
        return []

    min_count = select.get("minCount", 0)
    max_count = select.get("maxCount", 1)

    # sanity check to make sure max and min are actually the maximum and minimum amount of options available
    max_count = min(max_count, len(options))
    min_count = min(min_count, max_count)

    # CABT option type IDs from the API docs:
    # PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, DISCARD=11,
    # RETREAT=12, ATTACK=13, END=14
    END = 14
    ATTACK = 13

    # read option type
    def option_type(i: int):
        return options[i].get("type")

    all_indices = list(range(len(options)))

    # do the max amount of choice as possible
    if max_count > 1:
        return all_indices[:max_count]

    # If only one choice, try to do a non attacking/non end choice
    non_attack_non_end = [i for i in all_indices if option_type(i) not in (ATTACK, END)]

    if non_attack_non_end:
        return [non_attack_non_end[0]]

    # If no other actions exist, attack if possible.
    attack_options = [i for i in all_indices if option_type(i) == ATTACK]

    if attack_options:
        return [attack_options[0]]

    # If nothing else exists, end the turn.
    end_options = [i for i in all_indices if option_type(i) == END]

    if end_options:
        return [end_options[0]]

    # Choose the first option (however should not reach here)
    return [0]
