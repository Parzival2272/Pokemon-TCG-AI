"""Build deck.csv from lucario_deck.csv.

lucario_deck.csv holds the resolved 60 card IDs (source of truth).
deck.csv is the runtime file consumed by agent.py and the cabt engine.

    python build_deck.py
"""

from __future__ import annotations

from pathlib import Path

LUCARIO_DECK = Path(__file__).parent / "lucario_deck.csv"
DECK_OUT = Path(__file__).parent / "deck.csv"


def read_ids(path: Path) -> list[int]:
    """Parse a deck CSV: one card ID per non-blank line, no header."""
    return [int(line) for line in path.read_text().splitlines() if line.strip()]


def load_ids(path: Path = LUCARIO_DECK) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Source deck not found: {path}")
    ids = read_ids(path)
    if len(ids) != 60:
        raise ValueError(f"Expected 60 cards in {path}, got {len(ids)}")
    return ids


def write_deck(ids: list[int], path: Path = DECK_OUT) -> None:
    path.write_text("".join(f"{cid}\n" for cid in ids))


def main() -> None:
    ids = load_ids()
    write_deck(ids)
    print(f"Wrote {len(ids)} card IDs from {LUCARIO_DECK.name} to {DECK_OUT}")


if __name__ == "__main__":
    main()
