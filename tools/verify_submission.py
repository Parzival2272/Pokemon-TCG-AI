"""Verify a submission tarball actually runs the way the Kaggle sandbox runs it.

Extracts the bundle to a temp directory and plays full battles there in a
child process whose sys.path contains ONLY the extracted bundle -- so a file
that was left out of the tarball fails here instead of failing on Kaggle --
and with torch / sb3_contrib / stable_baselines3 blocked at import, since the
sandbox does not have them (agent.py runs the policy on plain NumPy for
exactly this reason).

Checks, in order:
  1. `import main` succeeds with only the bundle on sys.path.
  2. No forbidden module got imported along the way.
  3. The bundle's deck.csv matches the deck the policy trained on.
  4. main.agent plays complete battles, and every selection it returns is
     legal: indices in range, count within the engine's min/max, no dupes.

Usage:
    python tools/verify_submission.py [submission.tar.gz] [--games 2]
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORBIDDEN = ("torch", "sb3_contrib", "stable_baselines3")

# Runs inside the extracted bundle, with only that directory importable.
CHILD = r'''
import builtins, json, os, sys

# -I strips cwd from sys.path, so put the extracted bundle back -- and ONLY
# the bundle, which is the point of the check.
sys.path.insert(0, os.getcwd())

FORBIDDEN = %(forbidden)r
_real_import = builtins.__import__


def _guard(name, *a, **kw):
    if name.split(".")[0] in FORBIDDEN:
        raise ModuleNotFoundError(f"No module named {name!r}")  # as Kaggle would
    return _real_import(name, *a, **kw)


builtins.__import__ = _guard

import main  # noqa: E402
from ptcg.game import battle_finish, battle_select, battle_start  # noqa: E402

result = {"deck": main._deck, "games": [], "decisions": 0, "active": main.ACTIVE_AGENT}

for game in range(%(games)d):
    obs, start = battle_start(main._deck.copy(), main._deck.copy())
    if start.errorPlayer >= 0:
        raise SystemExit(f"engine rejected the bundled deck (player {start.errorPlayer})")
    steps = 0
    while True:
        current = obs.get("current") or {}
        if current.get("result", -1) >= 0:
            result["games"].append({"steps": steps, "result": current["result"]})
            break
        select = obs.get("select") or {}
        options = select.get("option") or []
        n = len(options)
        lo = min(select.get("minCount", 1), n)
        hi = min(select.get("maxCount", 1), n)

        picks = main.agent(obs)  # the real submission entry point
        result["decisions"] += 1

        # Exactly the contract the engine enforces on a submission.
        if not isinstance(picks, list) or not all(isinstance(p, int) for p in picks):
            raise SystemExit(f"agent returned {picks!r}, not list[int]")
        if len(set(picks)) != len(picks):
            raise SystemExit(f"agent returned duplicate indices: {picks!r}")
        if any(p < 0 or p >= n for p in picks):
            raise SystemExit(f"agent returned out-of-range index: {picks!r} (n={n})")
        if not (lo <= len(picks) <= hi):
            raise SystemExit(f"agent returned {len(picks)} picks, want {lo}..{hi}")

        obs = battle_select(picks)
        steps += 1
        if steps > 5000:
            raise SystemExit("battle did not terminate within 5000 selections")
    battle_finish()

imported = sorted(m for m in sys.modules if m.split(".")[0] in FORBIDDEN)
result["forbidden_imported"] = imported
print("RESULT " + json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tarball", nargs="?", default=os.path.join(ROOT, "submission.tar.gz"))
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()

    with open(os.path.join(ROOT, "crustle_deck.csv")) as f:
        training_deck = [int(line) for line in f if line.strip()]

    with tempfile.TemporaryDirectory(prefix="verify-sub-") as staged:
        with tarfile.open(args.tarball) as tar:
            tar.extractall(staged, filter="data")

        env = dict(os.environ)
        # Only the bundle is importable: no repo root, no inherited PYTHONPATH.
        env.pop("PYTHONPATH", None)
        env["PYTHONSAFEPATH"] = "1"

        proc = subprocess.run(
            [sys.executable, "-I", "-c", CHILD % {"forbidden": FORBIDDEN, "games": args.games}],
            cwd=staged,
            env=env,
            capture_output=True,
            text=True,
        )

    noise = [l for l in proc.stdout.splitlines() if not l.startswith("RESULT ")]
    payload = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    if proc.returncode != 0 or payload is None:
        print("\n".join(noise))
        print(proc.stderr, file=sys.stderr)
        sys.exit(f"FAILED (exit {proc.returncode})")

    result = json.loads(payload[len("RESULT "):])
    problems = []
    if result["forbidden_imported"]:
        problems.append(f"imported unavailable modules: {result['forbidden_imported']}")
    if result["deck"] != training_deck:
        problems.append("bundled deck.csv is NOT the training deck (crustle_deck.csv)")

    print(f"tarball:        {args.tarball}")
    print(f"active agent:   {result['active']}")
    print(f"deck.csv:       {'matches crustle_deck.csv' if result['deck'] == training_deck else 'MISMATCH'}")
    print(f"forbidden imports: {result['forbidden_imported'] or 'none'}")
    print(f"decisions made: {result['decisions']}")
    for i, game in enumerate(result["games"]):
        print(f"  game {i}: {game['steps']} selections, result={game['result']}")

    if problems:
        sys.exit("FAILED: " + "; ".join(problems))
    print("\nPASS")


if __name__ == "__main__":
    main()
