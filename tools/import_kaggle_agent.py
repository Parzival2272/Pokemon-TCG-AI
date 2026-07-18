"""Import a Kaggle rule-based agent script as a local agent package.

Usage:
    python tools/import_kaggle_agent.py <kaggle_script.py> <name> [--deck <deck.csv>] [--force]

Example:
    python tools/import_kaggle_agent.py ~/Downloads/abomasnow_notebook.py abomasnow

This creates heuristics/<name>_agent/ with:
    <name>_agent.py  - the script, with `cg.api` imports rewritten to `ptcg.api`
                       and the Kaggle deck-loading block replaced by a
                       package-local loader plus a set_deck() hook
    __init__.py      - re-exports agent and set_deck
    deck.csv         - built from decklist comments ("Kyogre = 721  # x2"),
                       or copied from --deck if given

and registers the new agent in main.py's AGENTS/SET_DECKS dicts. Switch to it
by setting ACTIVE_AGENT = "<name>" in main.py and copying
heuristics/<name>_agent/deck.csv over the project-root deck.csv (main.py
pushes the root deck.csv into the active agent, matching the Kaggle
submission layout).

The conversion targets the standard Kaggle starter layout (deck loading that
starts with `file_path = "deck.csv"`). If a script deviates, the tool warns
and you finish that part by hand -- always review the generated file.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"
HEURISTICS_DIR = PROJECT_ROOT / "heuristics"

LOADER_TEMPLATE = '''# Load deck.csv: package-local first, then the Kaggle agent directory.
import os as _os

_deck_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "deck.csv")
if not _os.path.exists(_deck_path):
    _deck_path = "/kaggle_simulations/agent/deck.csv"
with open(_deck_path) as _f:
    {var}: list[int] = [int(_line) for _line in _f.read().splitlines() if _line.strip()]


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py."""
    global {var}
    {var} = list(deck_list)
'''

INIT_TEMPLATE = """from .{module} import agent, set_deck

__all__ = ["agent", "set_deck"]
"""


def strip_notebook_magics(text: str) -> str:
    """Drop Jupyter magics such as the %%writefile header."""
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("%")]
    return "\n".join(lines).lstrip("\n") + "\n"


def rewrite_imports(text: str) -> str:
    """Point the Kaggle `cg` module imports at the local `ptcg` package."""
    text = re.sub(r"\bfrom cg\.api import\b", "from ptcg.api import", text)
    text = re.sub(r"\bfrom cg import\b", "from ptcg import", text)
    text = re.sub(r"\bimport cg\.api\b", "import ptcg.api", text)
    return text


def find_deck_var(text: str) -> str:
    """Detect the module-level deck variable (usually `my_deck`)."""
    m = re.search(r"^\s*(\w+)\.append\(int\(", text, re.M)
    if m:
        return m.group(1)
    m = re.search(r"^\s*(\w*deck\w*)\s*=\s*\[", text, re.M)
    if m:
        return m.group(1)
    return "my_deck"


def replace_deck_loader(text: str, var: str) -> tuple[str, bool]:
    """Swap the Kaggle deck-loading block for the package-local loader.

    The block is recognized from its first line (`file_path = "deck.csv"`) and
    consumed until the first top-level statement that is not part of loading
    the deck. Returns (new_text, replaced).
    """
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r'file_path\s*=\s*["\']deck\.csv["\']', ln.strip()):
            start = i
            break
    if start is None:
        return text, False

    loader_line = re.compile(
        r"^(file_path\b|if not os\.path\.exists|with open\(|csv\s*=|"
        rf"{re.escape(var)}\b|for \w+ in range)"
    )
    end = start
    while end < len(lines):
        ln = lines[end]
        if ln.strip() == "" or ln[:1] in (" ", "\t") or loader_line.match(ln):
            end += 1
        else:
            break
    # Give back trailing blank lines so spacing around the next block survives.
    while end > start and lines[end - 1].strip() == "":
        end -= 1

    # Preserve a leading comment such as "# Load deck.csv in the dataset".
    if start > 0 and lines[start - 1].strip().startswith("#"):
        start -= 1

    new_lines = lines[:start] + LOADER_TEMPLATE.format(var=var).splitlines() + lines[end:]
    return "\n".join(new_lines) + "\n", True


def parse_decklist(text: str) -> list[int]:
    """Build a deck from decklist comments like `Kyogre = 721  # x2`."""
    deck: list[int] = []
    for m in re.finditer(
        r"^([A-Za-z_]\w*)\s*=\s*(\d+)\s*#\s*[×xX]\s*(\d+)\s*$", text, re.M
    ):
        deck.extend([int(m.group(2))] * int(m.group(3)))
    return deck


def register_in_main(name: str, pkg: str) -> bool:
    """Add imports plus AGENTS/SET_DECKS entries to main.py. Idempotent."""
    text = MAIN_PY.read_text()
    if f"from heuristics.{pkg} import" in text:
        print(f"main.py: {pkg} already registered, skipping")
        return True

    # Imports go right after the last existing agent-package import.
    last = None
    for last in re.finditer(r"^from heuristics\.\w+_agent import .*\n", text, re.M):
        pass
    if last is None:
        return False
    text = (
        text[: last.end()]
        + f"from heuristics.{pkg} import agent as {name}_agent\n"
        + f"from heuristics.{pkg} import set_deck as {name}_set_deck\n"
        + text[last.end() :]
    )

    # New entry goes just before each dict's closing brace.
    entries = (("AGENTS", f"{name}_agent"), ("SET_DECKS", f"{name}_set_deck"))
    for dict_name, value in entries:
        idx = text.find(f"{dict_name} = {{")
        if idx == -1:
            return False
        close = text.index("\n}", idx)
        text = text[: close + 1] + f'    "{name}": {value},\n' + text[close + 1 :]

    MAIN_PY.write_text(text)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", type=Path, help="Kaggle agent script (.py)")
    ap.add_argument("name", help='short agent name, e.g. "abomasnow"')
    ap.add_argument(
        "--deck", type=Path, default=None,
        help="deck.csv to copy instead of parsing the decklist comments",
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing <name>_agent/"
    )
    args = ap.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.name):
        sys.exit(f"name must be a lowercase identifier, got {args.name!r}")
    pkg = f"{args.name}_agent"
    pkg_dir = HEURISTICS_DIR / pkg
    if pkg_dir.exists() and not args.force:
        sys.exit(f"{pkg_dir} already exists (use --force to overwrite)")

    text = strip_notebook_magics(args.source.read_text(encoding="utf-8"))
    warnings: list[str] = []

    if "def agent(" not in text:
        warnings.append("no `def agent(` found -- is this really an agent script?")

    text = rewrite_imports(text)
    var = find_deck_var(text)
    text, replaced = replace_deck_loader(text, var)
    if not replaced:
        warnings.append(
            "deck-loading block not recognized: add a package-local loader plus "
            "set_deck() by hand (see heuristics/crustle_agent/crustle_agent.py)"
        )

    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / f"{pkg}.py").write_text(text, encoding="utf-8")
    (pkg_dir / "__init__.py").write_text(INIT_TEMPLATE.format(module=pkg))

    if args.deck is not None:
        shutil.copyfile(args.deck, pkg_dir / "deck.csv")
        deck_note = f"copied from {args.deck}"
    else:
        deck = parse_decklist(text)
        (pkg_dir / "deck.csv").write_text("".join(f"{c}\n" for c in deck))
        deck_note = f"{len(deck)} cards parsed from decklist comments"
        if len(deck) != 60:
            warnings.append(
                f"deck.csv has {len(deck)} cards, not 60 -- fix heuristics/{pkg}/deck.csv "
                "by hand (or rerun with --deck path/to/deck.csv)"
            )

    if not register_in_main(args.name, pkg):
        warnings.append(
            "could not update main.py: add the imports and the AGENTS/SET_DECKS "
            f'entries for "{args.name}" yourself'
        )

    print(f"created {pkg_dir.relative_to(PROJECT_ROOT)}/ ({deck_note})")
    print(f'registered "{args.name}" in main.py')
    print(
        f'to play it: set ACTIVE_AGENT = "{args.name}" in main.py and copy '
        f"heuristics/{pkg}/deck.csv over deck.csv"
    )
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()
