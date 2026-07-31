"""Per-episode reward-term contributions under the ACTUAL v13 policy against
the real heuristic opponent pool -- the numbers that matter for judging whether
a repaired term is a nudge or a rival objective. (v13 itself was trained with
the broken terms; this is what the repaired shaping would have been paying it.)
"""
from collections import Counter, defaultdict

import numpy as np
from sb3_contrib import MaskablePPO

from training.cabt_env import CabtEnv
from training.rewards import REWARD_TERMS
from training.train import OPPONENT_POOL

N_EPISODES = 40

model = MaskablePPO.load("ppo_starmie_v13.zip", device="cpu")
env = CabtEnv(opponent_agents=OPPONENT_POOL)
obs, _ = env.reset(seed=11)

totals = defaultdict(float)
hits = Counter()
per_opp_turnends = []
episodes = 0
steps = 0
ep_terms = defaultdict(float)
results = Counter()

while episodes < N_EPISODES:
    mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
    obs, _, done, _, info = env.step(int(action))
    steps += 1
    for name, value in (info.get("reward_terms") or {}).items():
        totals[name] += value
        ep_terms[name] += value
        if value:
            hits[name] += 1
    if done:
        episodes += 1
        results[info.get("opponent", "?")] += 1
        per_opp_turnends.append(ep_terms.get("no_attack", 0.0))
        ep_terms.clear()
        obs, _ = env.reset()

print(f"{episodes} episodes / {steps} steps vs {dict(results)}\n")
print(f"{'term':18s} {'per episode':>12s} {'steps hit':>10s}   status")
for name in REWARD_TERMS:
    t = totals.get(name, 0.0) / episodes
    h = hits.get(name, 0)
    print(f"{name:18s} {t:12.4f} {h:10d}   {'LIVE' if h else 'never fired'}")
shaping = sum(totals[n] for n in REWARD_TERMS if n != "terminal") / episodes
print(f"\nshaping_total      {shaping:12.4f}")
print(f"terminal           {totals['terminal']/episodes:12.4f}")
print(f"\nno_attack per episode: min={min(per_opp_turnends):.2f} "
      f"median={sorted(per_opp_turnends)[len(per_opp_turnends)//2]:.2f} "
      f"max={max(per_opp_turnends):.2f}")
