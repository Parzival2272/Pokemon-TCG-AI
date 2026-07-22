"""Export a trained MaskablePPO policy's weights to a plain NumPy .npz file.

agent.py runs on these weights instead of loading the sb3_contrib .zip
directly, because Kaggle's submission sandbox does not have
torch/sb3_contrib/stable_baselines3 installed. Re-run this after every
retrain so the submission bundle matches the latest model.

Usage:
    python tools/export_policy_weights.py ppo_starmie_v2.zip ppo_starmie_v2_weights.npz
"""

import sys

import numpy as np
from sb3_contrib import MaskablePPO


def export(model_path: str, weights_path: str) -> None:
    model = MaskablePPO.load(model_path)
    sd = model.policy.state_dict()

    weights = {
        "w0": sd["mlp_extractor.policy_net.0.weight"].numpy(),
        "b0": sd["mlp_extractor.policy_net.0.bias"].numpy(),
        "w2": sd["mlp_extractor.policy_net.2.weight"].numpy(),
        "b2": sd["mlp_extractor.policy_net.2.bias"].numpy(),
        "wa": sd["action_net.weight"].numpy(),
        "ba": sd["action_net.bias"].numpy(),
    }
    np.savez(weights_path, **weights)

    print(f"Exported {model_path} -> {weights_path}")
    for k, v in weights.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python {sys.argv[0]} <model.zip> <weights.npz>")
    export(sys.argv[1], sys.argv[2])
