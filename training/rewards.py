def compute_reward(obs_dict, prev_obs_dict, done, result, your_index):
    reward = 0.0

    if done:
        return 1.0 if result != your_index else -1.0

    current = obs_dict.get("current") or {}
    prev_current = prev_obs_dict.get("current") or {}
    players = current.get("players") or [{}, {}]
    prev_players = prev_current.get("players") or [{}, {}]

    you = players[your_index]
    opp = players[1 - your_index]
    prev_opp = prev_players[1 - your_index]

    # Reward for taking opponent prizes
    opp_prizes_now = len(opp.get("prize") or [])
    opp_prizes_before = len(prev_opp.get("prize") or [])
    if opp_prizes_now < opp_prizes_before:
        reward += 0.2 * (opp_prizes_before - opp_prizes_now)

    return reward
