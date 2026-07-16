from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.obs_vectorizer import MAX_DECK_SLOTS, MAX_HAND, obs_to_vector, set_vectorizer_deck


PRIZE_START = 4 + 4 + (4 * 5) + (4 * 5) + MAX_HAND
PRIZE_END = PRIZE_START + MAX_DECK_SLOTS


def prize_block(obs_dict: dict) -> np.ndarray:
    vec = obs_to_vector(obs_dict)
    return vec[PRIZE_START:PRIZE_END]


def print_block(title: str, block: np.ndarray) -> None:
    zero_indices = np.where(block == 0.0)[0].tolist()
    one_indices = np.where(block == 1.0)[0].tolist()
    print(title)
    print(f"  zeros: {len(zero_indices)} -> {zero_indices}")
    print(f"  ones:  {len(one_indices)} -> {one_indices}")
    print(f"  block: {block.tolist()}")


def make_base_obs(deck_count: int, prize_count: int = 6) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "hand": [],
                    "discard": [],
                    "prize": [None] * prize_count,
                    "deckCount": deck_count,
                },
                {
                    "active": [],
                    "bench": [],
                    "hand": [],
                    "discard": [],
                    "prize": [None] * prize_count,
                    "deckCount": 60,
                },
            ],
            "stadium": [],
            "looking": None,
        },
        "select": {
            "context": 0,
            "minCount": 0,
            "maxCount": 1,
            "option": [],
            "deck": [],
        },
    }


def test_prize_boolean_no_reveal() -> None:
    deck = list(range(1, 61))
    set_vectorizer_deck(deck)

    obs_dict = make_base_obs(deck_count=60)
    block = prize_block(obs_dict)

    print("Test: prize boolean with no reveal")
    print_block("  expected: all 1.0", block)
    assert np.all(block == 1.0), "All slots should be 1.0 when nothing has been revealed"
    print("  ✓ passed\n")


def test_prize_boolean_partial_reveal() -> None:
    deck = list(range(1, 61))
    set_vectorizer_deck(deck)

    obs_dict = make_base_obs(deck_count=54)
    obs_dict["current"]["players"][0]["hand"] = [{"id": 7}, {"id": 21}]
    obs_dict["select"]["deck"] = [
        {"id": 5, "serial": 1, "playerIndex": 0},
        {"id": 10, "serial": 2, "playerIndex": 0},
        {"id": 15, "serial": 3, "playerIndex": 0},
    ]

    block = prize_block(obs_dict)

    print("Test: prize boolean with partial reveal")
    print_block("  expected: visible cards are 0.0, hidden cards are 1.0", block)

    for idx in [4, 6, 9, 14, 20]:
        assert block[idx] == 0.0, f"Deck slot {idx} should be 0.0 after reveal/hand visibility"
    assert block[0] == 1.0, "Unseen slots should remain 1.0"
    print("  ✓ passed\n")


def test_prize_boolean_full_deck_reveal() -> None:
    deck = list(range(1, 61))
    set_vectorizer_deck(deck)

    revealed_cards = [{"id": cid, "serial": i + 1, "playerIndex": 0} for i, cid in enumerate(deck[:54])]
    obs_dict = make_base_obs(deck_count=54)
    obs_dict["select"]["deck"] = revealed_cards

    block = prize_block(obs_dict)

    print("Test: prize boolean with full deck reveal")
    print_block("  expected: 54 visible slots are 0.0, 6 hidden prize slots are 1.0", block)

    for idx in range(54):
        assert block[idx] == 0.0, f"Deck slot {idx} should be 0.0 after full reveal"
    for idx in range(54, 60):
        assert block[idx] == 1.0, f"Deck slot {idx} should stay 1.0 because it was not revealed"
    print("  ✓ passed\n")


def test_prize_boolean_duplicates() -> None:
    deck = list(range(1, 57)) + [99, 99, 99, 99]
    set_vectorizer_deck(deck)

    obs_dict = make_base_obs(deck_count=57)
    
    obs_dict["current"]["players"][0]["hand"] = [{"id": 99}]
    obs_dict["current"]["players"][0]["discard"] = [{"id": 99}]
    obs_dict["select"]["deck"] = [
        {"id": 99, "serial": 1, "playerIndex": 0}
    ]

    block = prize_block(obs_dict)

    print("Test: prize boolean with duplicate cards (4-ofs)")
    print_block("  expected: three 99s are 0.0, one 99 is 1.0. Rest are 1.0.", block)

    duplicate_slots = block[56:60]
    
    zeros = np.sum(duplicate_slots == 0.0)
    ones = np.sum(duplicate_slots == 1.0)
    
    assert zeros == 3, f"Expected exactly three '99' slots to be 0.0, got {zeros}"
    assert ones == 1, f"Expected exactly one '99' slot to be 1.0 (prized), got {ones}"
    
    assert np.all(block[:56] == 1.0), "Unseen unique cards should all remain 1.0"
    print("  ✓ passed\n")


def test_prize_boolean_deep_board_state() -> None:
    deck = list(range(1, 61))
    set_vectorizer_deck(deck)

    obs_dict = make_base_obs(deck_count=52)
    
    obs_dict["current"]["players"][0]["active"] = [{
        "id": 5,
        "energyCards": [{"id": 6}, {"id": 7}],
        "tools": [{"id": 8}],
        "preEvolution": [{"id": 9}]
    }]
    
    obs_dict["current"]["players"][0]["bench"] = [{
        "id": 15,
        "energyCards": [{"id": 16}]
    }]
    
    obs_dict["current"]["stadium"] = [{"id": 20, "playerIndex": 0}]
    obs_dict["current"]["players"][0]["discard"] = [{"id": 25}]

    block = prize_block(obs_dict)

    print("Test: prize boolean with deep nested board state")
    print_block("  expected: Active, bench, energies, tools, stadium, and discard are 0.0", block)

    # Calculate expected 0-based indices from the IDs
    expected_visible_indices = [
        4, 5, 6, 7, 8,   # Active (ID 5) + Energies (6,7) + Tool (8) + PreEvo (9)
        14, 15,          # Bench (ID 15) + Energy (16)
        19,              # Stadium (ID 20)
        24               # Discard (ID 25)
    ]

    for idx in expected_visible_indices:
        assert block[idx] == 0.0, f"Deck slot {idx} should be 0.0 (found on board)"
        
    assert np.sum(block == 0.0) == 9, "Exactly 9 specific cards should be visible"
    print("  ✓ passed\n")

if __name__ == "__main__":
    try:
        test_prize_boolean_no_reveal()
        test_prize_boolean_partial_reveal()
        test_prize_boolean_full_deck_reveal()
        test_prize_boolean_duplicates()
        test_prize_boolean_deep_board_state()
        
        print("=" * 60)
        print("ALL BOOLEAN PRIZE TESTS PASSED ✓")
        print("=" * 60)
    except AssertionError as exc:
        print(f"\n✗ TEST FAILED: {exc}")
        sys.exit(1)