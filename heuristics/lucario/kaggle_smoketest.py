"""Self-contained Kaggle smoke test for the Lucario Hariyama deck.

Run inside a Kaggle notebook attached to the `pokemon-tcg-ai-battle`
competition.

  - Builds deck.csv from lucario_deck.csv (source of truth).
  - Runs a self-play game with kaggle_environments.make("cabt").
  - Saves /kaggle/working/replay.json (HTML render needs cabt.js).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from kaggle_environments import make

from agent import agent
from build_deck import DECK_OUT, LUCARIO_DECK as DECK_SRC, load_ids, write_deck

KAGGLE_DECK = Path("/kaggle/working/deck.csv")


def main(render_html: str | None = "/kaggle/working/replay.html") -> None:
    ids = load_ids(DECK_SRC)
    write_deck(ids, DECK_OUT)
    print(f"Built {DECK_OUT} from {DECK_SRC} ({len(ids)} cards)")
    print(f"First 10 IDs: {ids[:10]}")

    if KAGGLE_DECK.parent.exists():
        shutil.copy(DECK_OUT, KAGGLE_DECK)
        print(f"Copied deck to {KAGGLE_DECK}")

    print("\nRunning self-play game...")
    env = make("cabt", configuration={"decks": [ids.copy(), ids.copy()]}, debug=True)
    env.run([agent, agent])

    final = env.state
    print(f"\nSteps played: {len(env.steps)}")
    print(f"Final per-agent state: {final}")
    if final:
        rewards = [s.get("reward") for s in final]
        statuses = [s.get("status") for s in final]
        print(f"Rewards: {rewards}   Statuses: {statuses}")

    if render_html:
        json_path = Path(render_html).with_suffix(".json")
        json_path.write_text(json.dumps(env.toJSON(), indent=2, default=str))
        print(f"Game JSON written to {json_path}")

        try:
            Path(render_html).write_text(env.render(mode="html"))
            print(f"HTML replay written to {render_html}")
        except FileNotFoundError as exc:
            print(f"(HTML render unavailable: {exc.filename} missing — JSON dump is the replay)")


if __name__ == "__main__":
    main()
