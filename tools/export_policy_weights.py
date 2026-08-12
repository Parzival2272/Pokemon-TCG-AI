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
    # device="cpu" is required, not a preference: MaskablePPO.load() otherwise
    # restores onto CUDA when it is available, and every .numpy() below then
    # raises "can't convert cuda:0 device type tensor to numpy". The export is
    # pure tensor copying, so there is nothing to gain from the GPU anyway.
    model = MaskablePPO.load(model_path, device="cpu")
    sd = model.policy.state_dict()

    # Every Linear in the policy tower, discovered rather than hardcoded.
    # SB3 lays mlp_extractor.policy_net out as Linear/activation pairs, so the
    # Linear layers sit at even indices -- but how MANY there are is
    # POLICY_NET_ARCH's depth, which is not fixed.
    #
    # This used to name layers 0 and 2 literally. That was correct only while
    # POLICY_NET_ARCH had exactly two entries, and it did not FAIL when that
    # stopped being true -- it silently exported the first two layers of a
    # deeper net and dropped the rest, producing a weights file that loads
    # fine and plays as a different (wrong) network. With the arch now at
    # [512, 512, 512] that would have quietly shipped a truncated policy.
    prefix = "mlp_extractor.policy_net."
    depths = sorted(
        int(k[len(prefix):].split(".")[0])
        for k in sd
        if k.startswith(prefix) and k.endswith(".weight")
    )
    if not depths:
        raise RuntimeError(
            f"No {prefix}*.weight tensors in {model_path}; the policy is not "
            "the MlpPolicy tower this exporter (and agent.py) expect."
        )

    weights = {}
    for layer_index, sd_index in enumerate(depths):
        weights[f"w{layer_index}"] = sd[f"{prefix}{sd_index}.weight"].numpy()
        weights[f"b{layer_index}"] = sd[f"{prefix}{sd_index}.bias"].numpy()
    # n_layers is what lets agent.py loop the right number of times instead of
    # assuming a depth. Older exports predate it; agent.py falls back to the
    # legacy w0/w2 naming when it is absent.
    weights["n_layers"] = np.array(len(depths))
    weights["wa"] = sd["action_net.weight"].numpy()
    weights["ba"] = sd["action_net.bias"].numpy()
    np.savez(weights_path, **weights)

    print(f"Exported {model_path} -> {weights_path}")
    print(f"  policy tower: {len(depths)} hidden layers")
    for k, v in weights.items():
        print(f"  {k}: {v.shape} {v.dtype}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: python {sys.argv[0]} <model.zip> <weights.npz>")
    export(sys.argv[1], sys.argv[2])
