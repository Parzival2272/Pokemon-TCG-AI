"""Build the Kaggle submission tarball.

Bundles exactly what the sandbox needs to run main.py's ACTIVE_AGENT and
nothing else: the entry points, the policy weights .npz, the observation
vectorizer, the engine bindings, and every heuristic package main.py imports.

Two rules this script exists to enforce, both of which the previous
hand-rolled bundle got wrong:

  * deck.csv in the bundle is a copy of DECK_PATH (crustle_deck.csv) -- the
    deck the policy was TRAINED on, not heuristics/crustle_agent/deck.csv.
    Those two have drifted by four cards, and agent.py computes the
    observation's 60-feature prize-belief block against whichever one it
    loads, so shipping the wrong one silently degrades every inference.
  * the heuristic package list is derived from main.py's imports rather than
    hand-maintained. The previous bundle omitted alakazam_v2_agent after it
    was added to main.py, which is an ImportError on the first call in the
    sandbox -- a submission that fails every game.

The sandbox has no torch/sb3_contrib/stable_baselines3 (see agent.py), so
nothing here may pull them in; verify_submission.py checks that.

Usage:
    python tools/export_policy_weights.py ppo_crustle_v2.zip \
        models/ppo_crustle_v2_weights.npz
    python tools/build_submission.py            # -> submission.tar.gz
    python tools/build_submission.py --out foo.tar.gz
"""

import argparse
import ast
import os
import re
import shutil
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Engine bindings. Kaggle runs Linux (libcg.so); cg.dll is carried along so an
# extracted bundle also runs on a Windows dev box.
PTCG_FILES = ["__init__.py", "api.py", "game.py", "sim.py", "utils.py", "cg.dll", "libcg.so"]


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def required_heuristics():
    """Heuristic package names imported by main.py, in sorted order."""
    names = set()
    for node in ast.walk(ast.parse(_read("main.py"))):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("heuristics."):
            names.add(node.module.split(".")[1])
    return sorted(names)


def active_weights():
    """The weights filename agent.py defaults to (its PPO_WEIGHTS fallback)."""
    match = re.search(r'PPO_WEIGHTS",\s*"([^"]+)"', _read("agent.py"))
    if not match:
        sys.exit("Could not find the PPO_WEIGHTS default in agent.py")
    return match.group(1)


def training_deck():
    """DECK_PATH from training/cabt_env.py -- the deck the policy trained on."""
    match = re.search(r'^DECK_PATH\s*=\s*"([^"]+)"', _read("training", "cabt_env.py"), re.M)
    if not match:
        sys.exit("Could not find DECK_PATH in training/cabt_env.py")
    return match.group(1)


def build(out_path):
    weights = active_weights()
    deck_src = training_deck()
    heuristics = required_heuristics()

    weights_src = os.path.join(ROOT, "models", weights)
    if not os.path.exists(weights_src):
        sys.exit(
            f"{weights_src} does not exist -- run tools/export_policy_weights.py "
            "on the trained .zip first."
        )

    staged = tempfile.mkdtemp(prefix="submission-")
    try:
        for name in ("main.py", "agent.py"):
            shutil.copy2(os.path.join(ROOT, name), os.path.join(staged, name))

        # The bundle's deck.csv is the TRAINING deck, whatever the repo root
        # copy happens to hold.
        shutil.copy2(os.path.join(ROOT, deck_src), os.path.join(staged, "deck.csv"))

        os.makedirs(os.path.join(staged, "models"))
        shutil.copy2(weights_src, os.path.join(staged, "models", weights))

        # training/ has no __init__.py in this repo -- it resolves as a
        # namespace package, so shipping the one module agent.py needs is
        # enough (and keeps train.py's torch/sb3 imports out of the bundle).
        os.makedirs(os.path.join(staged, "training"))
        shutil.copy2(
            os.path.join(ROOT, "training", "obs_vectorizer.py"),
            os.path.join(staged, "training", "obs_vectorizer.py"),
        )

        os.makedirs(os.path.join(staged, "ptcg"))
        for name in PTCG_FILES:
            shutil.copy2(
                os.path.join(ROOT, "ptcg", name), os.path.join(staged, "ptcg", name)
            )

        os.makedirs(os.path.join(staged, "heuristics"))
        shutil.copy2(
            os.path.join(ROOT, "heuristics", "__init__.py"),
            os.path.join(staged, "heuristics", "__init__.py"),
        )
        for pkg in heuristics:
            shutil.copytree(
                os.path.join(ROOT, "heuristics", pkg),
                os.path.join(staged, "heuristics", pkg),
                ignore=shutil.ignore_patterns("__pycache__"),
            )

        with tarfile.open(out_path, "w:gz") as tar:
            for entry in sorted(os.listdir(staged)):
                tar.add(os.path.join(staged, entry), arcname=entry)
    finally:
        shutil.rmtree(staged, ignore_errors=True)

    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print(f"  weights:    models/{weights}")
    print(f"  deck.csv:   copied from {deck_src}")
    print(f"  heuristics: {', '.join(heuristics)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "submission.tar.gz"))
    build(parser.parse_args().out)
