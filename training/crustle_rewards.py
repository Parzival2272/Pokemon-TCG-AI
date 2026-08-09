"""Reward shaping for the RL learner -- Mega Kangaskhan ex / Crustle list.

Same contract as the Mega Starmie ex version of this file: rewards are always
from the perspective of the player who took the step's action ("me", an
absolute player index), the base signal is the sparse terminal +1 win / -1
loss, and on non-terminal steps we add dense shaping. `compute_reward` is the
single-value entry point; `reward_terms` returns the named breakdown for
per-component TensorBoard logging.

THE GAMEPLAN, AND THE ONE MATCHUP THAT INVERTS IT
-------------------------------------------------
Default line: Mega Kangaskhan ex (300 HP, {C}, Basic) parks in the Active Spot
drawing 2 a turn off Run Errand and does nothing else, while TWO Crustle get
built on the bench. Crustle (150 HP, Stage 1) has Mysterious Rock Inn -- it
takes zero damage from attacks by opposing Pokemon ex -- and swings for 120
with Superb Scissors at {G}{C}{C}, ignoring effects on the defender. Growing
Grass Energy is +20 HP each and Hero's Cape +100, so a dressed Crustle is a
250-330 HP wall an ex deck cannot damage at all. Building that wall is THE
priority; Kangaskhan is a draw engine and a body, not an attacker.

The shaping is therefore deliberately lopsided. Energy onto the Crustle line
and Hero's Cape onto Crustle pay large; energy onto Kangaskhan, Cape onto
Kangaskhan, and attacking with Kangaskhan are all charged, hard.

The exception is Alakazam (MEG 56, id 743 -- and its Abra/Kadabra line). It is
a Stage 2, NOT a Pokemon ex, so Mysterious Rock Inn does nothing and the wall
plan is dead. Powerful Hand *places damage counters* for each card in their
hand, which is an effect of an attack rather than damage -- and Mist Energy
prevents all effects of attacks on its holder. So against Alakazam the correct
line inverts completely: Mist Energy onto Kangaskhan as fast as possible, then
swing with Kangaskhan behind a blanked attack. Every Kangaskhan penalty flips
sign in that matchup and the Mist bonus is multiplied.

Two more matchups get named Boss's Orders targets:
  * Iono's Bellibolt ex: drag up Iono's Voltorb (the setup Basic) when Crustle
    can finish it, Iono's Kilowattrel second.
  * Mega Lucario ex: a CHARGED Hariyama first -- it is the piece that can
    already swing into Kangaskhan's x2 Fighting weakness right now -- then
    Makuhita (cuts the evolution line off before it becomes that threat), then
    a bare Hariyama last.
And several named priorities that apply regardless of matchup: Froslass TWM
53, the Crustle mirror, and (Abomasnow matchup) their benched Kyogre. Froslass's
Freezing Shroud drops a counter on every Pokemon with an Ability each Checkup,
which chips our Crustles straight through Rock Inn -- Battle Cage (new in this
list) blanks it, so the stadium term pays extra for dropping it in that
matchup, and Boss's Orders ALWAYS prioritizes dragging a Froslass up over
anything else on their bench, named or not (it shows up as a small tech in
other lists too, Grimmsnarl included, not just its own line). In the mirror,
the Boss priority is an opposing Crustle that hasn't reached attack cost yet
-- it can't hit back, so it eats free damage every turn it's forced to stay
Active. Against Abomasnow, Boss always goes after the benched Kyogre with the
most Energy on it -- see BOSS_ABOMASNOW_KYOGRE_REWARD for the caveat that this
one is running on unverified card IDs.

VERIFIED VS INFERRED
--------------------
Card IDs come from Card_ID_List_EN.pdf and card text from the printed cards;
those are solid. Observation-schema assumptions are inherited from the starmie
file, which documents which were confirmed against real obs["logs"] dumps:
Evolve/Attach/HpChange/Draw/Attack entry shapes and the LogType-is-an-IntEnum
finding are confirmed. Everything else here is marked INFERRED at its helper
and fails safe to 0 rather than raising. Terms that log a flat 0.0 for a whole
run are broken, not inactive -- that is what REWARD_TERMS exists to expose.

DECKLIST NOTE: Pokemon Center Lady (MEG 123) was cut -- the engine's card list
has no id for it (MEG 120/121/122/124/126 are present, 123 and 125 are absent)
-- and the slot is now Festival Grounds. That makes the Stadium the deck's ONLY
answer to Special Conditions: it makes every Pokemon in play with Energy
attached recover from, and stay immune to, Special Conditions. Confusion on a
Kangaskhan that is holding the Active Spot is the case that actually costs
turns, so the stadium term pays heavily for that specific play.
"""

from collections import Counter


# ── Global dial ───────────────────────────────────────────────────────────
# Multiplies every non-terminal term. The repeatable-per-turn terms are the
# ones to watch in the logs (damage, energy_attach/target, run_errand,
# wall_matchup, kangaskhan_attack); one-shots (lead, setup_tempo,
# crustle_pair) and once-per-copy card terms are bounded by the decklist and
# cannot run away. If the agent starts farming shaping instead of winning,
# drop this to 0.5 before touching individual constants.
SHAPING_SCALE = 1.0

# ── Core terms ────────────────────────────────────────────────────────────
PRIZE_REWARD = 0.1
MULTI_PRIZE_MULTIPLIER = 1.25
DAMAGE_REWARD_PER_100 = 0.01
SUPPORTER_PLAY_REWARD = 0.005
# NOTE: there is deliberately no per-card draw reward. It was double-paying
# for the Kangaskhan engine that `run_errand` already prices, and rewarding
# raw card flow pushed against the deck-out term.

# ── Energy attachment ─────────────────────────────────────────────────────
ENERGY_ATTACH_REWARD = 0.004
ENERGY_BONUS_BY_ID = {}  # populated after the IDs below

# Building the bench Crustle is the whole plan, so the target matters more
# than the attachment itself. Energy on Kangaskhan is charged even when no
# Crustle is waiting for it -- at reduced rate, since fuelling the only body
# we have is not senseless, but it is still not the plan.
# A benched Crustle outranks an Active one as an attach target: the bench is
# where the big wall gets assembled while Kangaskhan draws, and energy put on
# the Active is energy that isn't stacking HP on the thing we plan to promote.
# When nothing on the bench still wants energy, the Active is a fine target.
ENERGY_ON_BENCH_CRUSTLE_REWARD = 0.018
ENERGY_ON_ACTIVE_CRUSTLE_REWARD = 0.012
ENERGY_ON_ACTIVE_CRUSTLE_WHEN_BENCH_WANTS_PENALTY = 0.010

# Outside the Alakazam matchup there are exactly three boards on which energy
# belongs on Kangaskhan, and the penalties below are sized so the difference
# is unmistakable rather than a rounding error next to the attach bonus:
#   1. we can convert it into an immediate knockout  -> small reward
#   2. there is no Crustle line in play at all       -> neutral, it is the
#      only body we have; the real fix is finding a Dwebble, which is what
#      TUTOR_NO_LINE_REWARD pays for
#   3. anything else                                 -> heavy charge
ENERGY_ON_KANGASKHAN_PENALTY = 0.120          # a line member still wants it
ENERGY_ON_KANGASKHAN_LINE_FULL_PENALTY = 0.030  # line exists but is maxed out
ENERGY_ON_KANGASKHAN_NO_LINE_PENALTY = 0.0    # nothing better to attach to
ENERGY_ON_KANGASKHAN_FOR_KO_REWARD = 0.020

# Ordering. The attachment is once per turn, so spending it on Kangaskhan
# before developing a body forecloses the better target for the whole turn:
# with a Poffin in hand and an empty board, Poffin -> Dwebble -> attach is
# strictly better than attach -> Poffin -> Dwebble. The reward function never
# sees the order of actions inside a turn, but it does see what was still
# sitting in hand at the moment of the attach, which is enough to tell the two
# sequences apart. Charged instead of the neutral no-line rate.
ENERGY_ON_KANGASKHAN_MISORDERED_PENALTY = 0.080

# ...all of which inverts against Alakazam: there Kangaskhan IS the attacker
# and Mist Energy is what makes it unkillable by Powerful Hand.
ENERGY_ON_KANGASKHAN_VS_ALAKAZAM_REWARD = 0.020

# Mist onto Kangaskhan against Alakazam is the single highest-value non-
# terminal event in the file, by request: it is the play that decides the
# matchup outright, since Powerful Hand only ever places damage counters and
# Mist prevents all attack effects on its holder.
#
# BUDGET WARNING, read before tuning up. Terminal is +/-1.0 and a whole game's
# prizes are 0.6. With the scaling below, a game where all four Mist are found
# and attached pays roughly 0.55 from attaches plus up to 0.5 from the two
# Hilda -- i.e. Alakazam-matchup shaping is already close to the magnitude of
# winning. Raise ALAKAZAM_MIST_EMPHASIS if you want it higher still, but watch
# the reward/mist_vs_alakazam and reward/hilda_mist series against
# reward/terminal: if the agent starts chasing Mist in positions where it
# should be closing the game out, this is the dial that did it.
ALAKAZAM_MIST_EMPHASIS = 1.0
MIST_ON_KANGASKHAN_VS_ALAKAZAM_REWARD = 0.250
# Per-copy falloff. ONE Mist already blanks Powerful Hand completely -- copies
# 2-4 are insurance against the holder being knocked out or a fresh attacker
# needing cover, which is real but worth less. This also bounds the total.
MIST_COPY_SCALING = (1.0, 0.7, 0.5, 0.35)

# ── Development ───────────────────────────────────────────────────────────
EVOLVE_REWARD = 0.005
CRUSTLE_EVOLVE_REWARD = 0.060   # raised: too few Crustle were getting made
CRUSTLE_PAIR_REWARD = 0.070     # one-shot, the first time TWO Crustle are out

# Hero's Cape belongs on Crustle and essentially nowhere else. The only board
# where it is correct on Kangaskhan is Alakazam, where Kangaskhan attacks.
HERO_CAPE_ON_CRUSTLE_REWARD = 0.120
# The Cape is +100 HP -- it can never convert into a knockout, so the "easy KO"
# exemption that applies to energy does NOT apply here. Outside Alakazam there
# is no board where putting the single copy on Kangaskhan is right, including
# when no Crustle is out yet: it should be held for the one that arrives.
HERO_CAPE_ON_KANGASKHAN_PENALTY = 0.150
HERO_CAPE_ON_KANGASKHAN_VS_ALAKAZAM_REWARD = 0.060
HERO_CAPE_WASTED_PENALTY = 0.040

# Petrel tutors any Trainer, and with a single Cape in the list, fetching it
# is one of the highest-value things Petrel does.
PETREL_FETCHES_CAPE_REWARD = 0.060
# When the board has no Crustle-line member AND the hand has no Dwebble,
# Poffin or Ultra Ball either, Petrel (fetching a Poffin) is the ONLY card
# left that can put a body on the board at all -- every other Supporter is
# strictly worse in that exact spot. Stacks on top of the normal tutor payout.
PETREL_LAST_RESORT_REWARD = 0.050

# ── Healing ───────────────────────────────────────────────────────────────
# Jumbo Ice Cream heals 80 off an Active with 3+ energy. Fixed charge plus a
# per-10-HP bonus, so it breaks even near 35 HP and is best at the full 80.
ICE_CREAM_BASE = -0.020
ICE_CREAM_PER_10HP = 0.006
HEAL_REWARD_PER_10HP = 0.003    # Community Center and anything else

# ── Resource denial ───────────────────────────────────────────────────────
HAMMER_PLAY_REWARD = 0.004
HAMMER_LAND_REWARD = 0.020
# Target choice. Stripping the Active sets back the Pokemon that is attacking
# us right now and can force a retreat they may not be able to pay; a benched
# Pokemon has the whole turn cycle to get re-fuelled before it matters. The
# bench charge only applies when their Active actually had Energy on it to
# take instead -- a bench hit is fine when it was the only legal target.
HAMMER_ACTIVE_TARGET_REWARD = 0.020
HAMMER_BENCH_TARGET_PENALTY = 0.010

XEROSIC_PER_CARD_REWARD = 0.012   # cuts them to 3
XEROSIC_MAX_REWARD = 0.080

# Hand Trimmer cuts BOTH players to 5, opponent first -- so it is only good
# when their hand is fat and ours is already lean. The self-discard is priced.
TRIMMER_PER_CARD_REWARD = 0.012
TRIMMER_MAX_REWARD = 0.060
TRIMMER_SELF_DISCARD_PENALTY = 0.010

# ── Boss's Orders targeting ───────────────────────────────────────────────
# Bumped above every other named target (even a charged Hariyama plus its KO
# bonus, 0.160) on purpose: Froslass shows up as a small tech in more decks
# than just its own line (Grimmsnarl included), and whenever it's on their
# bench it should always win the Boss pick, matchup or not.
BOSS_FROSLASS_REWARD = 0.170        # Freezing Shroud chips through Rock Inn
BOSS_VOLTORB_REWARD = 0.100         # Bellibolt matchup, primary
BOSS_KILOWATTREL_REWARD = 0.070     # Bellibolt matchup, secondary
# Lucario matchup, in order. A Hariyama with Energy on it is the piece that
# can actually threaten a Crustle, so it outranks cutting the line off at
# Makuhita; a bare Hariyama is the least urgent of the three. Mega Lucario ex
# itself is deliberately NOT a target: Mysterious Rock Inn means it cannot
# damage Crustle at all, so dragging it up trades a killable threat for one we
# can neither hurt nor be hurt by.
BOSS_HARIYAMA_CHARGED_REWARD = 0.110
BOSS_MAKUHITA_REWARD = 0.090
BOSS_HARIYAMA_UNCHARGED_REWARD = 0.060
# Extra on top of a named threat's constant when it is ALSO inside Superb
# Scissors range -- i.e. this Boss converts straight into a knockout next
# turn, not just an improved Active Spot.
BOSS_NAMED_KO_BONUS = 0.050
# Crustle mirror: a Crustle with less than attack-cost Energy can't hit back
# at all if we drag it to Active, so it eats free damage every turn it stays
# there. Sized close to a charged Hariyama since an unpunishable target is
# just as valuable as a killable one.
BOSS_MIRROR_LOW_ENERGY_REWARD = 0.150
# Abomasnow matchup: always take their benched Kyogre with the most Energy on
# it, full HP or not. UNVERIFIED -- ABOMASNOW and KYOGRE ids below are not
# confirmed against Card_ID_List_EN.pdf the way the rest of this file's ids
# are, so this branch is a best-effort placeholder; fix the ids before relying
# on it in a real run.
BOSS_ABOMASNOW_KYOGRE_REWARD = 0.120
BOSS_KO_RANGE_REWARD = 0.030        # generic: inside Superb Scissors
BOSS_MULTI_PRIZE_REWARD = 0.030
# Fallback ordering when nothing named is available: weakest target first,
# then whatever is most expensive for them to retreat back out of.
BOSS_WEAK_TARGET_MAX_REWARD = 0.020
BOSS_HIGH_RETREAT_REWARD = 0.015
BOSS_WASTED_PENALTY = 0.020
# Boss is scored as an IMPROVEMENT on their Active Spot, not on the new target
# in isolation. Dragging up something no better than what was already standing
# there spends the turn's Supporter to accomplish nothing -- or worse, swaps a
# threat we could kill for one we cannot. This also covers giving up a FREE
# knockout: if the Pokemon already Active was inside KO range and Boss dragged
# up a fresh full-HP body instead, that is always a downgrade, named target or
# not -- see the guard in _boss_reward.
BOSS_DOWNGRADE_PENALTY = 0.060

# ── Tempo / targeting ─────────────────────────────────────────────────────
# Switch only pays when it actually lands a *built* Crustle in the Active
# Spot. The old version rewarded any swap out of a stuck Active, which the
# agent was cashing on turn 1 by Switching Kangaskhan away for nothing. The
# payout scales with how much Energy the landed Crustle is carrying (half at
# attack cost, full at CRUSTLE_MAX_USEFUL_ENERGY) so a board with two ready
# Crustles on the bench pulls the agent toward the fuller one, not just
# whichever one clears the "built" bar first.
SWITCH_TO_BUILT_CRUSTLE_REWARD = 0.040
# Any other Switch -- one that does NOT land a built Crustle Active -- is a
# card spent for nothing, whether that's shuffling Kangaskhan out at random or
# swapping into an under-charged bench mon. The old version scored this 0.0,
# which is exactly what let Switch get burned at random with nothing pulling
# it back. Charged much harder in the mirror, where Switch is one of the
# highest-value cards in the list (see SWITCH_KANGASKHAN_TO_CRUSTLE_MIRROR_
# REWARD below), so wasting it there is a bigger relative mistake.
SWITCH_WASTED_PENALTY = 0.015
SWITCH_WASTED_MIRROR_PENALTY = 0.040
# Crustle mirror: pulling a damaged/dead-weight Mega Kangaskhan ex out of the
# Active Spot and dropping in a Crustle that can swing right into it is close
# to the best single card play the deck has in that matchup -- Rock Inn is a
# no-op mirror vs mirror, so the game is decided by who lands the first free
# hits, and Switch does it instantly with no retreat cost paid and no attack
# given up.
SWITCH_KANGASKHAN_TO_CRUSTLE_MIRROR_REWARD = 0.100

POKEGEAR_WANTED_REWARD = 0.020
POKEGEAR_ANY_REWARD = 0.005

# Softened from (0.060, 0.010, -0.030): the old breakeven at a 6-card hand was
# harsh enough that ordinary mid-game Lillie plays (not empty-handed, not
# hoarding) scored negative on average, which showed up as a persistent
# negative trend on reward/lillie even though nothing was actually broken.
LILLIE_BASE_REWARD = 0.070
LILLIE_PER_CARD_DISCARDED = 0.008
LILLIE_MAX_PENALTY = -0.020

# Under deck-out pressure Lillie's value inverts. It shuffles the hand back in
# and draws 6, so the net change to the deck is (hand_size - 6): on a fat hand
# it is a deck REFILL, and on a lean one it burns 6 more cards we cannot spare.
# The reward is capped at LILLIE_SWEET_SPOT_HAND on purpose -- past that point
# there is no gradient pushing the agent to hoard a bigger and bigger hand,
# which would just be handing free value to an opposing Xerosic's Machinations
# (it cuts us to 3 regardless of how many we were sitting on).
LILLIE_DECKOUT_THRESHOLD = 12
LILLIE_DECK_GAIN_PER_CARD = 0.015
LILLIE_DECKOUT_MAX_REWARD = 0.080
LILLIE_SWEET_SPOT_HAND = 9
LILLIE_DECKOUT_EARLY_PENALTY = 0.030

TUTOR_ON_NEED_REWARD = 0.020
# An empty board is the one situation where energy on Kangaskhan is forgiven,
# so the pressure has to land somewhere -- here, on actually going and getting
# a Dwebble instead of settling in behind Kangaskhan.
TUTOR_NO_LINE_REWARD = 0.050
# Hilda gets a Pokemon AND an Energy off a single card; Ultra Ball only ever
# gets the Pokemon, and pays for it by discarding 2 more cards on top. Hilda
# was being under-used relative to Ultra Ball, so this tips the choice toward
# Hilda whenever both are legal options.
HILDA_TUTOR_BONUS = 0.020

# Hilda searches out an Evolution Pokemon AND an Energy card in one play,
# which makes it strictly better than Ultra Ball whenever both are live
# options -- HILDA_TUTOR_BONUS above prices that comparison in. The Evolution
# half can only ever fetch Crustle in this list (Dwebble is a Basic, not an
# Evolution), and the Energy half is the only on-demand way to get either Mist
# Energy (the whole Alakazam matchup) or Growing Grass Energy (HP now, attack
# cost eventually) instead of top-decking one.
HILDA_CRUSTLE_REWARD = 0.070
HILDA_GRASS_ENERGY_REWARD = 0.035
# Against Alakazam the Energy half is the whole matchup: Powerful Hand places
# damage counters, Mist blanks attack effects on its holder, and we want all
# four found and stuck on Kangaskhan as fast as possible. The reward decays
# per turn so "asap" is actually encoded rather than just "eventually", and it
# scales with how many Mist we still don't have -- the first one matters most.
HILDA_MIST_VS_ALAKAZAM_REWARD = 0.250
HILDA_MIST_DECAY_PER_TURN = 0.025
HILDA_MIST_MIN_REWARD = 0.100
HILDA_MIST_REWARD = 0.020          # any other matchup -- Mist is still good

# Choosing the replacement after a knockout: anything holding Mist Energy is
# the first choice on any board, then the biggest Crustle -- or Kangaskhan in
# the Alakazam matchup.
PROMOTION_BEST_REWARD = 0.050
PROMOTION_WRONG_PENALTY = 0.030

# ── Run Errand ────────────────────────────────────────────────────────────
RUN_ERRAND_MISS_PENALTY = 0.010
RUN_ERRAND_MIN_DRAWS = 2

# ── Wasted turns ──────────────────────────────────────────────────────────
# Passing the turn while holding an attacker that was ready to swing. This is
# deliberately NOT the old unconditional no-attack penalty, which punished the
# core gameplan: with Kangaskhan Active outside the Alakazam matchup, NOT
# attacking is correct -- it is a draw engine, and KANGASKHAN_ATTACK_PENALTY
# charges for swinging with it. So this fires only when the Pokemon we
# actually want attacking was ready and sat there anyway.
NO_ATTACK_WITH_READY_ATTACKER_PENALTY = 0.050

# ── Setup tempo ───────────────────────────────────────────────────────────
SETUP_TEMPO_BASE = 0.080
SETUP_TEMPO_DECAY_PER_TURN = 0.015

# ── Wall matchup ──────────────────────────────────────────────────────────
CRUSTLE_WALL_ATTACK_REWARD = 0.012

# Kangaskhan should not be attacking. It is a 3-prize body whose job is
# drawing, and every swing exposes it. Flat charge, plus more when a built
# Crustle was sitting right there -- inverted against Alakazam.
KANGASKHAN_ATTACK_PENALTY = 0.060
KANGASKHAN_ATTACK_WITH_CRUSTLE_READY_PENALTY = 0.030
KANGASKHAN_ATTACK_VS_ALAKAZAM_REWARD = 0.040

# ── Retreating ────────────────────────────────────────────────────────────
RETREAT_ENERGY_PENALTY = 0.015
# The good version of a retreat: Kangaskhan can't (or shouldn't be trusted to)
# swing and a ready Crustle is waiting on the bench, so paying the retreat
# cost to bring the attacker in is what the turn was for. Outranks the flat
# energy-dump charge above rather than just discounting it -- this is the play
# that transitions the game from "Kangaskhan draws" to "Crustle attacks".
RETREAT_TO_READY_CRUSTLE_REWARD = 0.060

# ── Deck-out ──────────────────────────────────────────────────────────────
DECK_LOW_THRESHOLD = 8
DECK_CRITICAL_THRESHOLD = 3
DECK_LOW_PENALTY = 0.040
DECK_CRITICAL_PENALTY = 0.120

# ── Board shape ───────────────────────────────────────────────────────────
MAX_KANGASKHAN_IN_PLAY = 1
# Three line members on board is fine in most matchups -- a spare Dwebble is
# cheap insurance against a prize trade, and only a fourth starts crowding the
# bench and feeding Boss targets. The cap is a ceiling, not a target; the
# positive pull toward exactly two built Crustle is CRUSTLE_PAIR_REWARD.
MAX_CRUSTLE_LINE_IN_PLAY = 3
EXTRA_KANGASKHAN_PENALTY = 0.060
EXTRA_CRUSTLE_LINE_PENALTY = 0.030
# Mega Lucario ONLY. Kangaskhan is Fighting x2 and worth 3 Prizes, so every
# extra copy that hits the board there is a 150-effective-HP gift. Outside
# that matchup there is no benching penalty at all -- MAX_KANGASKHAN_IN_PLAY
# already caps the board at one, and stacking a second discouragement on top
# was making the trained bot avoid a perfectly fine board state. The OPENING
# Kangaskhan is exempt and in fact correct even against Lucario: it is the 300
# HP body that buys time for the Crustles. The bench-only check gets that
# exemption for free, since a Basic played from hand can only ever go to the
# Bench and the setup Active is not a Bench slot.
KANGASKHAN_BENCHED_VS_LUCARIO_PENALTY = 0.120

LEAD_KANGASKHAN_REWARD = 0.050
LEAD_WRONG_PENALTY = 0.050

# ── Stadium ───────────────────────────────────────────────────────────────
STADIUM_PLAY_REWARD = 0.030
STADIUM_BUMP_REWARD = 0.050
# Battle Cage stops damage counters landing on our Bench from effects and
# Abilities -- i.e. it blanks Freezing Shroud while the Crustles build.
BATTLE_CAGE_VS_COUNTER_DECK_REWARD = 0.050
# Festival Grounds is the only Special Condition answer in the list. Anything
# with Energy attached recovers immediately and stays immune, so dropping it
# while a Confused Active is stuck there is worth far more than a normal
# stadium play. Note the effect is symmetric -- it un-sticks their Pokemon too
# -- which is why the payout is gated on us actually needing it right now.
FESTIVAL_GROUNDS_CLEARS_CONFUSION_REWARD = 0.090
FESTIVAL_GROUNDS_CLEARS_CONDITION_REWARD = 0.045

# ── Our card IDs (Card_ID_List_EN.pdf) ────────────────────────────────────
BASIC_GRASS_ENERGY = 1         # SVE 1   (list has no MEE printing)
MIST_ENERGY = 11               # TEF 161
SPIKY_ENERGY = 14              # JTG 159
GROW_GRASS_ENERGY = 18         # POR 86

DWEBBLE = 344                  # DRI 11
CRUSTLE = 345                  # DRI 12
MEGA_KANGASKHAN_EX = 756       # MEG 104

BUDDY_BUDDY_POFFIN = 1086      # TEF 144
HAND_TRIMMER = 1087            # TEF 150
CRUSHING_HAMMER = 1120         # SVI 168 (list has no POR printing)
ULTRA_BALL = 1121              # SVI 196 (list has no MEG printing)
POKEGEAR_3 = 1122              # SVI 186
SWITCH = 1123                  # SVI 194 (list has no MEG printing)
JUMBO_ICE_CREAM = 1147         # PFL 91
HEROS_CAPE = 1159              # TEF 152
BOSSS_ORDERS = 1182            # PAL 172 (list has no MEG printing)
XEROSICS_MACHINATIONS = 1197   # SFA 64
TEAM_ROCKETS_PETREL = 1219     # DRI 176
HILDA = 1225                   # WHT 84
LILLIES_DETERMINATION = 1227   # MEG 119
COMMUNITY_CENTER = 1242        # TWM 146
TEAM_ROCKETS_FACTORY = 1257    # DRI 173
FESTIVAL_GROUNDS = 1245        # TWM 149
BATTLE_CAGE = 1264             # PFL 85

CRUSTLE_LINE_IDS = (DWEBBLE, CRUSTLE)
STADIUM_IDS = (TEAM_ROCKETS_FACTORY, COMMUNITY_CENTER, FESTIVAL_GROUNDS, BATTLE_CAGE)

# ── Opponent card IDs used for matchup detection and Boss targeting ───────
SNORUNT_TWM = 103              # TWM 51
FROSLASS_TWM = 104             # TWM 53  -- Freezing Shroud
SNORUNT_ASC = 860              # ASC 46
ABRA_TWM = 109                 # TWM 80
ALAKAZAM_TWM = 245             # TWM 82
IONOS_VOLTORB = 265            # JTG 47
IONOS_BELLIBOLT_EX = 269       # JTG 53
IONOS_KILOWATTREL = 271        # JTG 55
MAKUHITA = 673                 # MEG 72
HARIYAMA = 674                 # MEG 73
MEGA_LUCARIO_EX = 678          # MEG 77
ABRA_MEG = 741                 # MEG 54
KADABRA_MEG = 742              # MEG 55
ALAKAZAM_MEG = 743             # MEG 56  -- Powerful Hand

# PLACEHOLDER IDS -- NOT VERIFIED against Card_ID_List_EN.pdf. Known only as
# "Mega Abomasnow ex" and "Kyogre", both from the Mega Evolution (ME01) set;
# the collector number in that set does not tell us the engine's internal
# card id (compare MEGA_KANGASKHAN_EX = 756 for "MEG 104" -- no relation
# between the two numbers). Replace these two with the real ids before
# trusting BOSS_ABOMASNOW_KYOGRE_REWARD or MATCHUP_ABOMASNOW in a real run;
# until then both fail safe (an id that matches nothing just never fires).
MEGA_ABOMASNOW_EX = 9001       # ME01 -- Mega Abomasnow ex, UNVERIFIED id
KYOGRE_ME01 = 9002             # ME01 -- Kyogre, UNVERIFIED id

FROSLASS_LINE_IDS = (FROSLASS_TWM, SNORUNT_TWM, SNORUNT_ASC)
ALAKAZAM_LINE_IDS = (ALAKAZAM_MEG, KADABRA_MEG, ABRA_MEG, ALAKAZAM_TWM, ABRA_TWM)
BELLIBOLT_LINE_IDS = (IONOS_BELLIBOLT_EX, IONOS_VOLTORB, IONOS_KILOWATTREL)
LUCARIO_LINE_IDS = (MEGA_LUCARIO_EX, MAKUHITA, HARIYAMA)
ABOMASNOW_LINE_IDS = (MEGA_ABOMASNOW_EX,)

# The deck's own line, for mirror-matchup detection -- their Crustle can hurt
# ours (neither side is a Pokemon ex, so Rock Inn is a no-op mirror vs
# mirror), which flips Switch and Boss's Orders into some of the highest-
# value cards in the list. See _detect_matchup and the mirror branches in
# _switch_reward / _boss_target_value.
MIRROR_LINE_IDS = (MEGA_KANGASKHAN_EX, CRUSTLE, DWEBBLE)

MATCHUP_GENERIC = "generic"
MATCHUP_ALAKAZAM = "alakazam"
MATCHUP_MIRROR = "mirror"
MATCHUP_BELLIBOLT = "bellibolt"
MATCHUP_LUCARIO = "lucario"
MATCHUP_ABOMASNOW = "abomasnow"

# Superb Scissors damage, for "can Crustle finish this?" checks. Ignores
# Weakness and Resistance, same approximation the starmie file made.
# Rapid-Fire Combo: {C}{C}{C} for 200+ (coin-flip chain on top). The base 200
# is what the easy-KO check uses, so the exemption is conservative -- it never
# assumes the flips.
KANGASKHAN_ATTACK_COST = 3
KANGASKHAN_ATTACK_DAMAGE = 200

CRUSTLE_ATTACK_DAMAGE = 120
CRUSTLE_ATTACK_COST = 3        # {G}{C}{C} -- also crustle_agent.py's threshold
# Two ceilings, not one. CRUSTLE_MAX_USEFUL_ENERGY (5) is the "still chasing
# HP" ceiling -- Growing Grass Energy is +20 HP a copy, so it is worth loading
# a Crustle that hasn't found ANY grass yet all the way up to it. But once a
# Crustle already has grass on it, attack cost (3) is the real cap: it can
# already swing, and topping it off with anything else is no longer the
# priority. See _crustle_energy_cap.
CRUSTLE_MAX_USEFUL_ENERGY = 5
# What an attach that lands ABOVE that Crustle's cap is worth. Still positive
# -- a spare battery on the wall is never a real mistake -- just far below the
# normal placement reward, so the agent stops treating every attach as equally
# good once the line already has what it needs.
CRUSTLE_ENERGY_OVER_CAP_REWARD = 0.003

ENERGY_BONUS_BY_ID = {
    MIST_ENERGY: 0.012,        # blanks attack effects on the holder
    GROW_GRASS_ENERGY: 0.008,  # +20 HP each on the Grass wall
    SPIKY_ENERGY: 0.005,       # 2 counters back on whatever hits our Active
    BASIC_GRASS_ENERGY: 0.001,
}

try:
    from ptcg.api import all_card_data, CardType
    CARD_DB = {c.cardId: c for c in all_card_data()}
except Exception:
    CardType = None
    CARD_DB = {}

# Log entry "type" is an IntEnum (ptcg.api.LogType), NOT the string names
# vis.json renders -- see the starmie file's comment on the multi-million step
# runs that silently logged 0.0 because of this. Numeric fallbacks are the
# documented ptcg/api.py values.
try:
    from ptcg.api import LogType as _LogType
    LOG_DRAW = int(_LogType.DRAW)
    LOG_ATTACH = int(_LogType.ATTACH)
    LOG_EVOLVE = int(_LogType.EVOLVE)
    LOG_ATTACK = int(_LogType.ATTACK)
    LOG_HP_CHANGE = int(_LogType.HP_CHANGE)
except Exception:
    LOG_DRAW, LOG_ATTACH, LOG_EVOLVE, LOG_ATTACK, LOG_HP_CHANGE = 4, 11, 12, 15, 16


# ── Shared accessors ──────────────────────────────────────────────────────

def _players(obs_dict):
    return (obs_dict.get("current") or {}).get("players") or [{}, {}]


def _player(obs_dict, player_index):
    players = _players(obs_dict)
    return (players[player_index] or {}) if player_index < len(players) else {}


def _energies(mon):
    """Energy CARDS attached to a Pokemon dict.

    "energyCards" is the authority, not "energies". Both keys are always
    present on a board Pokemon and always the same length, but they hold
    different things:

        {"id": 756, "energies": [0, 1, 0],
         "energyCards": [{"id": 14, ...}, {"id": 18, ...}, {"id": 11, ...}]}

    "energies" is a list of energy TYPE enums (0 Colorless, 1 Grass, 5
    Psychic); only "energyCards" carries real Card_ID_List_EN ids. This
    function used to prefer "energies" -- a guess from crustle_agent.py's
    typed `Pokemon.energies` -- which silently zeroed every id-aware Energy
    branch in this file, since `_energy_id(e) == MIST_ENERGY` was comparing a
    type enum against card id 11 and could never be true. That is why
    reward/mist_vs_alakazam was exactly 0.0 at all 872 logged points of
    MaskablePPO_22's 25M steps, and why ENERGY_BONUS_BY_ID never paid out.
    The rest of the repo already had this right (obs_vectorizer.py,
    rewards.py, the baa/ heuristics all read energyCards for ids and
    `len(energies)` for counts).

    Because the two lists are the same length, the many `len(_energies(...))`
    call sites (attack cost, energy caps, Boss target ranking) were correct
    either way and are unaffected by this change. Only the id reads --
    _has_mist, _mist_secured, _crustle_energy_cap's grass check,
    _board_snapshot/_energy_gains, and everything downstream of them -- change
    behavior.

    energyCards wins even when empty; a Pokemon with no attached Energy has
    both lists empty, so falling through to "energies" there could only
    resurrect the type-enum form. "energies"/"energy" remain as fallbacks for
    an observation that lacks energyCards entirely, where a correct count is
    still better than nothing (ids read off that path are still type enums and
    will not match any card constant). Returns [] rather than raising.
    """
    if not isinstance(mon, dict):
        return []
    cards = mon.get("energyCards")
    if cards is not None:
        return list(cards)
    for key in ("energies", "energy"):
        val = mon.get(key)
        if val:
            return list(val)
    return []


def _has_special_condition(mon, keyword=None):
    """INFERRED field name for Special Conditions -- pass keyword="CONFUS" for
    Confusion specifically, or None for any condition. Fails safe to False,
    which makes the Festival Grounds bonus inert rather than wrong. Pin this to
    the real field once you can dump a Confused Pokemon from a live obs."""
    if not isinstance(mon, dict):
        return False
    for key in ("conditions", "specialConditions", "status", "condition"):
        val = mon.get(key)
        if not val:
            continue
        candidates = val if isinstance(val, (list, tuple, set)) else [val]
        for c in candidates:
            name = (getattr(c, "name", None) or str(c)).upper()
            if keyword is None or keyword in name:
                return True
    return False


def _energy_id(energy):
    """Card id of one attached-Energy entry. The obs serializes an attached
    Energy either as a card dict or as a bare card id -- both occur in the same
    run -- so accept either and fail safe to None for anything else. Every read
    of an attached Energy's id must go through this; `(e or {}).get("id")`
    crashes the worker on the bare-int form (AttributeError in _board_snapshot,
    which kills the SubprocVecEnv pipe)."""
    if isinstance(energy, dict):
        return energy.get("id")
    if isinstance(energy, int) and not isinstance(energy, bool):
        return energy
    return getattr(energy, "id", None)


def _has_mist(mon):
    return any(_energy_id(e) == MIST_ENERGY for e in _energies(mon))


def _retreat_cost(mon):
    """INFERRED field name. Used only to rank Boss targets, so 0 (unknown)
    simply drops the high-retreat bonus rather than misranking anything."""
    if not isinstance(mon, dict):
        return 0
    for key in ("retreatCost", "retreat", "retreatCount"):
        val = mon.get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, (list, tuple)):
            return len(val)
    return 0


def _slot_key(mon, area, index):
    """Stable identifier for one board slot across a step. Prefers `serial`,
    which survives the bench reshuffling that follows a knockout; falls back to
    (area, index), which is fine within a single step where nothing was KO'd."""
    serial = mon.get("serial")
    return ("serial", serial) if serial is not None else (area, index)


def _board_snapshot(obs_dict, player_index):
    """{slot_key: (card_id, area, Counter(energy ids), maxHp)} for our side."""
    out = {}
    p = _player(obs_dict, player_index)
    for area, lst in (("active", p.get("active") or []), ("bench", p.get("bench") or [])):
        for i, m in enumerate(lst):
            if not m:
                continue
            out[_slot_key(m, area, i)] = (
                m.get("id"),
                area,
                Counter(_energy_id(e) for e in _energies(m)),
                m.get("maxHp"),
            )
    return out


def _energy_gains(prev_obs, cur_obs, player_index):
    """(receiving_pokemon_card_id, area, energy_card_id, energy_count_after,
    had_grass_before) for every Energy that appeared on one of our Pokemon
    this step.

    Derived by diffing the board, NOT by reading the Attach log. The log's
    `cardIdTarget` is an inferred field, and when it is absent every
    target-aware branch falls through to "unknown" and silently scores 0 --
    which is exactly the failure mode where the agent keeps loading Kangaskhan
    with no penalty ever landing. Diffing per-slot Energy counters needs only
    `active`, `bench` and `energies`, all of which are confirmed to exist.

    Also resolves bench-vs-active exactly, which the log could never do: with a
    Crustle in the Active Spot and another on the bench, cardIdTarget is the
    same value for both.

    energy_count_after and had_grass_before are running values within this
    step -- if more than one Energy landed on the same slot in one step (not
    expected under one-attachment-per-turn, but not assumed either), the
    second gain sees the first one already applied. That is what lets
    _energy_terms grade each attach against CRUSTLE_MAX_USEFUL_ENERGY /
    CRUSTLE_ATTACK_COST (see _crustle_energy_cap) without re-reading the
    board itself.
    """
    prev_map = _board_snapshot(prev_obs, player_index)
    cur_map = _board_snapshot(cur_obs, player_index)
    gains = []
    for key, (card_id, area, counter, _mhp) in cur_map.items():
        before = prev_map.get(key, (None, None, Counter(), None))[2]
        running_total = sum(before.values())
        had_grass = any(before.get(gid, 0) > 0
                         for gid in (GROW_GRASS_ENERGY, BASIC_GRASS_ENERGY))
        for energy_id, count in counter.items():
            added = count - before.get(energy_id, 0)
            for _ in range(max(0, added)):
                running_total += 1
                gains.append((card_id, area, energy_id, running_total, had_grass))
                if energy_id in (GROW_GRASS_ENERGY, BASIC_GRASS_ENERGY):
                    had_grass = True
    return gains


def _tool_gains(prev_obs, cur_obs, player_index):
    """Card ids of Pokemon whose maxHp jumped by >= 100 this step -- i.e. that
    just received a Hero's Cape (+100 HP).

    Same reasoning as _energy_gains: Tools do not show up in `energies`, and
    the Attach log's target field can't be relied on. Growing Grass Energy also
    raises maxHp, but only by 20 a copy, so a >= 100 jump can't be produced by
    energy alone within one step.
    """
    prev_map = _board_snapshot(prev_obs, player_index)
    cur_map = _board_snapshot(cur_obs, player_index)
    out = []
    for key, (card_id, area, _c, max_hp) in cur_map.items():
        before = prev_map.get(key, (None, None, Counter(), None))[3]
        if isinstance(max_hp, (int, float)) and isinstance(before, (int, float)):
            if max_hp - before >= 100:
                out.append((card_id, area))
    return out


def _pokemon_in_play(obs_dict, player_index):
    p = _player(obs_dict, player_index)
    return [m for m in (list(p.get("active") or []) + list(p.get("bench") or [])) if m]


def _bench_pokemon(obs_dict, player_index):
    return [m for m in (_player(obs_dict, player_index).get("bench") or []) if m]


def _active_pokemon(obs_dict, player_index):
    act = _player(obs_dict, player_index).get("active") or []
    return act[0] if act else None


def _count_in_play(obs_dict, player_index, card_id):
    return sum(1 for m in _pokemon_in_play(obs_dict, player_index) if m.get("id") == card_id)


def _count_any_in_play(obs_dict, player_index, card_ids):
    ids = set(card_ids)
    return sum(1 for m in _pokemon_in_play(obs_dict, player_index) if m.get("id") in ids)


def _hand_ids(obs_dict, player_index):
    return [c.get("id") for c in (_player(obs_dict, player_index).get("hand") or []) if c]


def _hand_size(obs_dict, player_index):
    """Cards in hand. OUR hand arrives as a real card list; the OPPONENT'S is
    hidden information, so their "hand" key is empty and len() on it is always
    0 -- which is what silently zeroed the Xerosic and Hand Trimmer terms.
    Falls back to whatever count field the observation exposes for a hidden
    hand. Returns 0 when nothing resolves, so callers must not treat 0 as
    "confirmed empty" -- prefer measuring the effect (see
    _opponent_cards_lost_from_hand) over reading this for the opponent."""
    p = _player(obs_dict, player_index)
    hand = p.get("hand")
    if hand:
        return len(hand)
    for key in ("handCount", "handSize", "handNum", "numHand"):
        val = p.get(key)
        if isinstance(val, int):
            return val
    return 0


def _opponent_cards_lost_from_hand(prev_obs, cur_obs, opp_index):
    """How many cards actually left the opponent's hand for their discard this
    step. Measures the EFFECT rather than the precondition, so it works
    against a hidden hand where _hand_size cannot. Used by the discard-based
    disruption terms; a Crushing Hammer resolving on the same step would also
    land here, which is a small over-count we accept rather than trying to
    attribute individual discarded cards."""
    return len(_newly_discarded_ids(prev_obs, cur_obs, opp_index))


def _discard_ids(obs_dict, player_index):
    return [c.get("id") for c in (_player(obs_dict, player_index).get("discard") or []) if c]


def _prizes_remaining(obs_dict, player_index):
    return len(_player(obs_dict, player_index).get("prize") or [])


def _deck_remaining(obs_dict, player_index):
    return len(_player(obs_dict, player_index).get("deck") or [])


def _newly_discarded_ids(prev_obs, cur_obs, player_index):
    """Card ids that entered player_index's discard since prev_obs (duplicates
    included). Count-based diff, so it assumes nothing about discard ordering,
    and it is a far safer "was this card played" detector than guessing a Play
    log schema. It also fires on non-play discards (Ultra Ball's cost, energy
    knocked off by Crushing Hammer) -- an accepted approximation, which is why
    the Hammer term cross-checks whose discard grew.
    """
    prev_ids = Counter(_discard_ids(prev_obs, player_index))
    cur_ids = Counter(_discard_ids(cur_obs, player_index))
    out = []
    for cid, cnt in cur_ids.items():
        out.extend([cid] * max(0, cnt - prev_ids.get(cid, 0)))
    return out


def _newly_in_hand_ids(prev_obs, cur_obs, player_index):
    """Card ids that appeared in our hand since prev_obs -- tells us what a
    search card actually fetched (Pokegear / Petrel / Hilda)."""
    prev_ids = Counter(_hand_ids(prev_obs, player_index))
    cur_ids = Counter(_hand_ids(cur_obs, player_index))
    out = []
    for cid, cnt in cur_ids.items():
        out.extend([cid] * max(0, cnt - prev_ids.get(cid, 0)))
    return out


# ── Card-type predicates ──────────────────────────────────────────────────
# CARD_DB first, then a hardcoded fallback covering our own 60. The fallback
# exists because every one of these gates a reward term, and with ptcg.api
# unimportable the CARD_DB-only version returns None for everything -- which
# silently zeroes the energy, supporter, Hammer and Ice Cream terms for the
# whole run. Terms that classify an *opponent's* card (Hammer's "did they lose
# an Energy") still need CARD_DB and fail safe to 0 without it.

DECK_ENERGY_IDS = frozenset({BASIC_GRASS_ENERGY, MIST_ENERGY, SPIKY_ENERGY, GROW_GRASS_ENERGY})
DECK_SUPPORTER_IDS = frozenset({
    LILLIES_DETERMINATION, BOSSS_ORDERS, TEAM_ROCKETS_PETREL, HILDA,
    XEROSICS_MACHINATIONS,
})
DECK_ITEM_IDS = frozenset({
    JUMBO_ICE_CREAM, POKEGEAR_3, BUDDY_BUDDY_POFFIN, ULTRA_BALL, SWITCH,
    CRUSHING_HAMMER, HAND_TRIMMER,
})


def _card_type(card_id):
    if not CARD_DB or card_id is None:
        return None
    data = CARD_DB.get(card_id)
    return getattr(data, "cardType", None) if data else None


def _classify(card_id, db_types, fallback_ids):
    ct = _card_type(card_id)
    if ct is not None and CardType is not None:
        return ct in db_types
    return card_id in fallback_ids


def _is_energy_card(card_id):
    return _classify(
        card_id,
        (getattr(CardType, "ENERGY", object()), getattr(CardType, "SPECIAL_ENERGY", object())),
        DECK_ENERGY_IDS,
    )


def _is_item_card(card_id):
    return _classify(card_id, (getattr(CardType, "ITEM", None),), DECK_ITEM_IDS)


def _is_supporter_card(card_id):
    return _classify(card_id, (getattr(CardType, "SUPPORTER", None),), DECK_SUPPORTER_IDS)


def _prize_value_by_id(card_id):
    if not CARD_DB or card_id is None:
        return 1
    data = CARD_DB.get(card_id)
    if data and getattr(data, "megaEx", False):
        return 3
    if data and getattr(data, "ex", False):
        return 2
    return 1


def _is_ex_pokemon(card_id):
    """CARD_DB-only: _prize_value_by_id defaults everything to 1 prize without
    the DB, so dependent terms go quiet rather than misfire."""
    return _prize_value_by_id(card_id) >= 2



# ── Log readers ───────────────────────────────────────────────────────────

def _logs(obs_dict):
    return obs_dict.get("logs") or []


def _entries(obs_dict, log_type, player_index):
    return [e for e in _logs(obs_dict)
            if isinstance(e, dict)
            and e.get("type") == log_type
            and e.get("playerIndex") == player_index]


def _draw_count(obs_dict, player_index):
    """Still used by run_errand even though there is no draw reward."""
    total = 0
    for e in _entries(obs_dict, LOG_DRAW, player_index):
        amount = e.get("count", e.get("amount"))
        total += amount if isinstance(amount, (int, float)) else 1
    return total


def _evolve_count(obs_dict, player_index):
    return len(_entries(obs_dict, LOG_EVOLVE, player_index))


def _hp_delta(obs_dict, player_index, positive):
    total = 0
    for e in _entries(obs_dict, LOG_HP_CHANGE, player_index):
        v = e.get("value")
        if isinstance(v, (int, float)) and ((v > 0) if positive else (v < 0)):
            total += v if positive else -v
    return total


def _damage_dealt(obs_dict, target_player_index):
    return _hp_delta(obs_dict, target_player_index, positive=False)


def _heal_dealt(obs_dict, player_index):
    return _hp_delta(obs_dict, player_index, positive=True)


def _attacker_id(obs_dict, player_index):
    """Card id of the Pokemon that attacked. `cardId` on Attack entries is the
    field the starmie agent reads for its Itchy Pollen check, so it exists;
    that it holds the attacker is INFERRED. Falls back to the Active."""
    entries = _entries(obs_dict, LOG_ATTACK, player_index)
    if not entries:
        return None
    cid = entries[0].get("cardId")
    if cid is not None:
        return cid
    mon = _active_pokemon(obs_dict, player_index)
    return mon.get("id") if mon else None


def _stadium_identity(obs_dict):
    stadium = (obs_dict.get("current") or {}).get("stadium") or []
    s = stadium[0] if stadium else None
    return (s.get("id"), s.get("playerIndex")) if s else None


# ── Board helpers specific to this deck ───────────────────────────────────

def _crustle_line_count(obs_dict, player_index):
    return _count_any_in_play(obs_dict, player_index, CRUSTLE_LINE_IDS)


def _best_crustle_energy(obs_dict, player_index):
    best = -1
    for m in _pokemon_in_play(obs_dict, player_index):
        if m.get("id") == CRUSTLE:
            best = max(best, len(_energies(m)))
    return best


def _built_crustle_count(obs_dict, player_index):
    return sum(1 for m in _pokemon_in_play(obs_dict, player_index)
               if m.get("id") == CRUSTLE and len(_energies(m)) >= CRUSTLE_ATTACK_COST)


def _ready_crustle_on_bench(obs_dict, player_index):
    return any(m.get("id") == CRUSTLE and len(_energies(m)) >= CRUSTLE_ATTACK_COST
               for m in _bench_pokemon(obs_dict, player_index))


def _ready_crustle_anywhere(obs_dict, player_index):
    return _built_crustle_count(obs_dict, player_index) > 0


def _crustle_energy_cap(mon):
    """How much Energy THIS Crustle-line body can still usefully hold.

    Growing Grass Energy is +20 HP a copy and keeps paying past attack cost,
    so a Crustle that hasn't found any grass yet is graded on the old, looser
    ceiling (CRUSTLE_MAX_USEFUL_ENERGY). One that already has grass on it is
    capped at attack cost (CRUSTLE_ATTACK_COST): it can already swing, and
    everything past that is a spare battery, not the plan.
    """
    has_grass = any(_energy_id(e) in (GROW_GRASS_ENERGY, BASIC_GRASS_ENERGY)
                     for e in _energies(mon))
    return CRUSTLE_MAX_USEFUL_ENERGY if not has_grass else CRUSTLE_ATTACK_COST


def _line_member_wants_energy(obs_dict, player_index):
    """Is there a Crustle-line body that could still use the attachment?

    Uses _crustle_energy_cap, not a flat threshold -- a Crustle sitting on
    exactly 3 with grass already attached is ready to swing AND capped, so
    treating it as still "wanting" energy is what let energy-onto-Kangaskhan
    escape the full penalty even after the line was actually served.
    """
    for m in _pokemon_in_play(obs_dict, player_index):
        if m.get("id") in CRUSTLE_LINE_IDS and len(_energies(m)) < _crustle_energy_cap(m):
            return True
    return False


def _bench_line_wants_energy(obs_dict, player_index):
    """Same test restricted to the Bench -- the thing that decides whether an
    attach to the Active Crustle was the wrong choice."""
    for m in _bench_pokemon(obs_dict, player_index):
        if m.get("id") in CRUSTLE_LINE_IDS and len(_energies(m)) < _crustle_energy_cap(m):
            return True
    return False


def _active_energy_count(obs_dict, player_index):
    return len(_energies(_active_pokemon(obs_dict, player_index)))


def _bench_energy_count(obs_dict, player_index):
    return sum(len(_energies(m)) for m in _bench_pokemon(obs_dict, player_index))


def _mist_secured(obs_dict, player_index):
    """Mist Energy we already hold or have attached. Counts hand + everything
    in play; Mist still sitting in the deck or prizes is what we're digging
    for."""
    count = sum(1 for cid in _hand_ids(obs_dict, player_index) if cid == MIST_ENERGY)
    for m in _pokemon_in_play(obs_dict, player_index):
        count += sum(1 for e in _energies(m) if _energy_id(e) == MIST_ENERGY)
    return count


def _could_have_developed_first(obs_dict, player_index, supporter_available):
    """Did we hold something that could have put a Crustle-line body on the
    board BEFORE spending the turn's attachment?

    Dwebble, Buddy-Buddy Poffin and Ultra Ball are all free actions -- benching
    a Basic or playing an Item costs nothing else in the turn, so holding one
    means the better sequence was available. Hilda and Petrel only count when
    the turn's Supporter hasn't been used yet, since otherwise they weren't
    actually playable.

    This does not check that the deck still contains a Dwebble to find; a
    Poffin with the line already prized or discarded will be treated as a
    missed development. Rare enough to accept over tracking deck contents.
    """
    hand = set(_hand_ids(obs_dict, player_index))
    if hand & {DWEBBLE, BUDDY_BUDDY_POFFIN, ULTRA_BALL}:
        return True
    if supporter_available and (hand & {HILDA, TEAM_ROCKETS_PETREL}):
        return True
    return False


def _active_should_attack(obs_dict, player_index, matchup):
    """Was our Active both able to attack and the thing we WANT attacking?

    Only two boards qualify, which keeps this consistent with the rest of the
    file rather than second-guessing it:
      * a Crustle at attack cost -- Superb Scissors is the gameplan
      * Kangaskhan at attack cost in the Alakazam matchup, where Rapid-Fire
        Combo is the gameplan
    A Kangaskhan in the Active Spot anywhere else is doing its job by NOT
    attacking, and Dwebble is deliberately excluded rather than guessing at an
    attack cost the card list doesn't give us.

    Asleep and Paralyzed Pokemon cannot attack at all, so they are exempt;
    Confusion is not, since a Confused Pokemon may still attack on a flip. That
    check rides on the same inferred condition field as Festival Grounds and
    fails safe to "no condition", which can only over-charge on the rare turn
    we were genuinely locked down.
    """
    active = _active_pokemon(obs_dict, player_index)
    if not active:
        return False
    if _has_special_condition(active, "ASLEEP") or _has_special_condition(active, "PARALY"):
        return False
    energy = len(_energies(active))
    if active.get("id") == CRUSTLE:
        return energy >= CRUSTLE_ATTACK_COST
    if matchup == MATCHUP_ALAKAZAM and active.get("id") == MEGA_KANGASKHAN_EX:
        return energy >= KANGASKHAN_ATTACK_COST
    return False


def _kangaskhan_can_take_ko(obs_dict, me_index, opp_index, energy_after_attach):
    """Is Kangaskhan one attachment away from simply knocking the defender out?

    Requires Kangaskhan to actually be the Active (Rapid-Fire Combo can't be
    used from the bench) and the defender to be inside the un-flipped 200. The
    energy count is the post-attach one, so this answers "does THIS attachment
    turn into a knockout", not "could it someday".
    """
    active = _active_pokemon(obs_dict, me_index)
    if not active or active.get("id") != MEGA_KANGASKHAN_EX:
        return False
    if energy_after_attach < KANGASKHAN_ATTACK_COST:
        return False
    defender = _active_pokemon(obs_dict, opp_index)
    if not defender:
        return False
    hp = defender.get("hp")
    return isinstance(hp, (int, float)) and hp <= KANGASKHAN_ATTACK_DAMAGE


def _stranded_dwebble(obs_dict, player_index):
    return (_count_in_play(obs_dict, player_index, DWEBBLE) > 0
            and CRUSTLE not in _hand_ids(obs_dict, player_index))


def _wants_ice_cream(obs_dict, player_index):
    mon = _active_pokemon(obs_dict, player_index)
    if not mon or len(_energies(mon)) < 3:
        return False
    hp, max_hp = mon.get("hp"), mon.get("maxHp")
    if not isinstance(hp, (int, float)) or not isinstance(max_hp, (int, float)):
        return False
    return (max_hp - hp) >= 60 and JUMBO_ICE_CREAM not in _hand_ids(obs_dict, player_index)


# ── Matchup detection ─────────────────────────────────────────────────────

def _detect_matchup(obs_dict, opp_index):
    """Sticky per game -- once a line is seen it stays identified, since the
    opponent's key Pokemon may be knocked out or benched later. Scans their
    board AND discard, so a Kadabra that already evolved still counts.

    Checked in priority order: Alakazam first because it is the one matchup
    that inverts the whole gameplan, so a mis-tag there is the costliest, then
    the mirror (their deck IS our deck, which flips several tempo cards).

    KNOWN GAP: Kadabra TWM (TWM 81, between ABRA_TWM and ALAKAZAM_TWM in that
    evolution line) has no id defined anywhere in this file -- unlike the MEG
    printing, which has all three stages. Evolution replaces the mon in place
    rather than discarding the lower stage, so a game where the TWM Alakazam
    line is sitting mid-evolution as Kadabra is invisible to `seen` until it
    evolves again, and matchup detection (and everything gated on
    MATCHUP_ALAKAZAM, including mist_vs_alakazam) stays MATCHUP_GENERIC until
    then. Add KADABRA_TWM's real id here once it's confirmed against
    Card_ID_List_EN.pdf -- do not guess it.

    ABOMASNOW_LINE_IDS is currently a single UNVERIFIED placeholder id (see
    its definition) and will not match anything in a real game until fixed.
    """
    seen = set(_discard_ids(obs_dict, opp_index))
    seen.update(m.get("id") for m in _pokemon_in_play(obs_dict, opp_index))
    if seen & set(ALAKAZAM_LINE_IDS):
        return MATCHUP_ALAKAZAM
    if seen & set(MIRROR_LINE_IDS):
        return MATCHUP_MIRROR
    if seen & set(ABOMASNOW_LINE_IDS):
        return MATCHUP_ABOMASNOW
    if seen & set(BELLIBOLT_LINE_IDS):
        return MATCHUP_BELLIBOLT
    if seen & set(LUCARIO_LINE_IDS):
        return MATCHUP_LUCARIO
    return MATCHUP_GENERIC


def _opponent_plays_counters(obs_dict, opp_index):
    """True when the opponent runs damage-counter effects that Battle Cage
    turns off on our Bench: Froslass's Freezing Shroud, Alakazam's Powerful
    Hand."""
    seen = set(_discard_ids(obs_dict, opp_index))
    seen.update(m.get("id") for m in _pokemon_in_play(obs_dict, opp_index))
    return bool(seen & (set(FROSLASS_LINE_IDS) | set(ALAKAZAM_LINE_IDS)))


def _wanted_supporter_ids(obs_dict, me_index, opp_index, matchup):
    """Supporters worth digging for right now -- used to grade Pokegear hits."""
    wanted = set()
    if _bench_pokemon(obs_dict, opp_index):
        wanted.add(BOSSS_ORDERS)
    if _stranded_dwebble(obs_dict, me_index) or _line_member_wants_energy(obs_dict, me_index):
        wanted.add(HILDA)
    # Against Alakazam, Hilda is the Mist tutor and stays wanted until all
    # four copies are accounted for.
    if matchup == MATCHUP_ALAKAZAM and _mist_secured(obs_dict, me_index) < 4:
        wanted.add(HILDA)
    if _hand_size(obs_dict, me_index) <= 3:
        wanted.add(LILLIES_DETERMINATION)
    # Their hand size is usually unreadable (hidden), so _hand_size returns 0
    # and this gate would never open. Treat the disruption Supporter as wanted
    # whenever we can't rule it out.
    if _hand_size(obs_dict, opp_index) >= 6 or _hand_size(obs_dict, opp_index) == 0:
        wanted.add(XEROSICS_MACHINATIONS)
    if (_crustle_line_count(obs_dict, me_index) < 2
            or _wants_ice_cream(obs_dict, me_index)
            or HEROS_CAPE not in _hand_ids(obs_dict, me_index)):
        wanted.add(TEAM_ROCKETS_PETREL)
    return wanted


# ── Per-game state ────────────────────────────────────────────────────────

_turns_taken = {0: 0, 1: 0}
_lead_scored = {0: False, 1: False}
_setup_scored = {0: False, 1: False}
_pair_scored = {0: False, 1: False}
_turn_draws = {0: 0, 1: 0}
_turn_kangaskhan_active = {0: False, 1: False}
_turn_supporter_used = {0: False, 1: False}
_turn_could_attack = {0: False, 1: False}
_matchup = {0: MATCHUP_GENERIC, 1: MATCHUP_GENERIC}


def reset_turn_tracking():
    """Call at the start of every game, both self-play and heuristic-opponent
    mode. Only correct for one game per process at a time -- key these by env
    id if games ever run concurrently through a shared process."""
    global _turns_taken, _lead_scored, _setup_scored, _pair_scored
    global _turn_draws, _turn_kangaskhan_active, _turn_supporter_used
    global _turn_could_attack, _matchup
    _turns_taken = {0: 0, 1: 0}
    _lead_scored = {0: False, 1: False}
    _setup_scored = {0: False, 1: False}
    _pair_scored = {0: False, 1: False}
    _turn_draws = {0: 0, 1: 0}
    _turn_kangaskhan_active = {0: False, 1: False}
    _turn_supporter_used = {0: False, 1: False}
    _turn_could_attack = {0: False, 1: False}
    _matchup = {0: MATCHUP_GENERIC, 1: MATCHUP_GENERIC}


reset_game_state = reset_turn_tracking


def _update_matchup(cur_obs, me_index, opp_index):
    """Sticky: once identified as something other than generic, it stays."""
    if _matchup.get(me_index, MATCHUP_GENERIC) == MATCHUP_GENERIC:
        _matchup[me_index] = _detect_matchup(cur_obs, opp_index)
    return _matchup[me_index]


def _turn_owner(obs_dict):
    """INFERRED: index of the player whose turn it currently is.
    `current.yourIndex` is the field the rest of this function assumes exists;
    if the engine names it differently, the turn-end branch below never fires
    and run_errand / no_attack silently stay at 0.0 for the entire run -- which
    is exactly the failure mode reward/run_errand showed. Falls back to
    sibling key names before giving up and returning None (which reproduces
    the old always-0 behavior rather than raising). Prefer confirming the real
    field against a live obs dump and dropping the fallbacks once you have."""
    current = obs_dict.get("current") or {}
    for key in ("yourIndex", "currentPlayerIndex", "activePlayerIndex",
                "turnPlayerIndex", "turnIndex"):
        val = current.get(key)
        if isinstance(val, int):
            return val
    return None


def _turn_end_penalties(prev_obs, cur_obs, me_index, matchup):
    """(run_errand, no_attack) -- both settle on turn hand-off, read off
    _turn_owner (current.yourIndex, with fallbacks) flipping away from us.

    run_errand: a turn that held Kangaskhan Active without drawing off Run
    Errand. Deliberately conservative -- draws from Lillie, Petrel or Team
    Rocket's Factory land in the same counter, so a turn that skipped the
    Ability but drew off a Supporter looks identical to one that used it. The
    threshold of 2 means we under-charge rather than punish turns that did.

    no_attack: passing the turn while an attacker we actually want swinging
    was ready (see _active_should_attack). Both conditions accumulate across
    every step of the turn, so a Crustle promoted or Switched in mid-turn
    still counts as having been available.

    Must be called exactly once per step: it mutates the accumulators and is
    what advances _turns_taken (which SETUP_TEMPO_DECAY_PER_TURN reads).
    """
    _turn_draws[me_index] = _turn_draws.get(me_index, 0) + _draw_count(cur_obs, me_index)
    active = _active_pokemon(prev_obs, me_index)
    if active and active.get("id") == MEGA_KANGASKHAN_EX:
        _turn_kangaskhan_active[me_index] = True
    if _active_should_attack(prev_obs, me_index, matchup):
        _turn_could_attack[me_index] = True

    if _turn_owner(cur_obs) == me_index:
        return 0.0, 0.0

    _turns_taken[me_index] = _turns_taken.get(me_index, 0) + 1
    missed_errand = (_turn_kangaskhan_active[me_index]
                     and _turn_draws[me_index] < RUN_ERRAND_MIN_DRAWS)
    attacked = bool(_entries(cur_obs, LOG_ATTACK, me_index))
    # From our second turn onward -- whoever goes first cannot attack on turn 1.
    wasted_turn = (_turn_could_attack[me_index]
                   and not attacked
                   and _turns_taken[me_index] >= 2)

    _turn_draws[me_index] = 0
    _turn_kangaskhan_active[me_index] = False
    _turn_supporter_used[me_index] = False
    _turn_could_attack[me_index] = False
    return (RUN_ERRAND_MISS_PENALTY if missed_errand else 0.0,
            NO_ATTACK_WITH_READY_ATTACKER_PENALTY if wasted_turn else 0.0)


def _lead_reward(prev_obs, cur_obs, me_index):
    """One-shot bonus for the opening Active being Kangaskhan -- true in every
    matchup, including Alakazam, where it is also the attacker. Grades the
    earliest board state this function sees.

    The wrong-lead penalty only fires if Kangaskhan was actually in the
    opening hand -- if it got prized instead, leading with anything else
    isn't a choice the agent had. Charging it anyway added an irreducible
    charge Kangaskhan-prized games via 6-card RNG that no policy improvement
    could ever remove, which is what showed up as a persistent negative floor
    on reward/lead rather than a real regression.
    """
    if _lead_scored.get(me_index):
        return 0.0
    mon = _active_pokemon(prev_obs, me_index) or _active_pokemon(cur_obs, me_index)
    if not mon:
        return 0.0
    _lead_scored[me_index] = True
    if mon.get("id") == MEGA_KANGASKHAN_EX:
        return LEAD_KANGASKHAN_REWARD
    hand = set(_hand_ids(prev_obs, me_index)) | set(_hand_ids(cur_obs, me_index))
    if MEGA_KANGASKHAN_EX not in hand:
        return 0.0
    return -LEAD_WRONG_PENALTY


# ── Term helpers ──────────────────────────────────────────────────────────

def _energy_terms(prev_obs, cur_obs, me_index, opp_index, matchup):
    """(attach_reward, target_reward, alakazam_bonus) for every Energy that
    landed on our board this step.

    Attachments and their targets come from _energy_gains -- a board diff --
    rather than from the Attach log, so the target-aware grading below cannot
    be silently disabled by a missing `cardIdTarget`. Every gain is classified,
    including ones landing on Pokemon we didn't expect, so there is no
    "unknown target" path that scores 0.

    Grading, outside Alakazam:
      bench Crustle line                        strong reward
      active Crustle line, bench still hungry    charged
      active Crustle line, bench full            small reward
      Kangaskhan, converts to a knockout         small reward
      Kangaskhan, no line in play at all         neutral
      Kangaskhan, line exists and wants energy   heavy charge
      Kangaskhan, line exists but maxed          charged
      anything else                              neutral

    Against Alakazam it inverts: Kangaskhan is the attacker, so energy on it is
    correct and Mist on it is the highest-value attachment in the file.
    """
    attach_reward = 0.0
    target_reward = 0.0
    zam_bonus = 0.0
    vs_zam = matchup == MATCHUP_ALAKAZAM

    has_line = _crustle_line_count(cur_obs, me_index) > 0
    line_available = _line_member_wants_energy(cur_obs, me_index)
    bench_wants = _bench_line_wants_energy(prev_obs, me_index)
    could_develop = _could_have_developed_first(
        prev_obs, me_index, not _turn_supporter_used.get(me_index, False))
    kang_energy_after = max(
        (len(_energies(m)) for m in _pokemon_in_play(cur_obs, me_index)
         if m.get("id") == MEGA_KANGASKHAN_EX),
        default=0,
    )
    mist_already = sum(
        sum(1 for e in _energies(m) if _energy_id(e) == MIST_ENERGY)
        for m in _pokemon_in_play(prev_obs, me_index)
        if m.get("id") == MEGA_KANGASKHAN_EX
    )

    for target_id, area, energy_id, energy_after, had_grass_before in _energy_gains(
            prev_obs, cur_obs, me_index):
        attach_reward += ENERGY_ATTACH_REWARD + ENERGY_BONUS_BY_ID.get(energy_id, 0.0)

        if target_id in CRUSTLE_LINE_IDS:
            # Mist is a fine defensive attach on a Crustle even outside its
            # normal role; against Alakazam specifically it is graded on the
            # looser ceiling rather than the tightened attack-cost one, since
            # Crustle is not the gameplan there and there's no reason to rush
            # the cap on it.
            if vs_zam and energy_id == MIST_ENERGY:
                cap = CRUSTLE_MAX_USEFUL_ENERGY
            else:
                cap = CRUSTLE_MAX_USEFUL_ENERGY if not had_grass_before else CRUSTLE_ATTACK_COST

            if energy_after > cap:
                placement = CRUSTLE_ENERGY_OVER_CAP_REWARD
            elif area == "bench":
                placement = ENERGY_ON_BENCH_CRUSTLE_REWARD
            elif bench_wants:
                # A bench Crustle was still hungry and we fed the Active one.
                placement = -ENERGY_ON_ACTIVE_CRUSTLE_WHEN_BENCH_WANTS_PENALTY
            else:
                placement = ENERGY_ON_ACTIVE_CRUSTLE_REWARD
            # Still worth something against Alakazam -- Crustle is a fine wall
            # against whatever else they promote -- just not the plan.
            target_reward += placement * (0.25 if vs_zam else 1.0)

        elif target_id == MEGA_KANGASKHAN_EX:
            if vs_zam:
                target_reward += ENERGY_ON_KANGASKHAN_VS_ALAKAZAM_REWARD
                if energy_id == MIST_ENERGY:
                    # Scaled by how many Mist Kangaskhan already had: the first
                    # one is the matchup-winning attachment.
                    scale = MIST_COPY_SCALING[min(mist_already, len(MIST_COPY_SCALING) - 1)]
                    zam_bonus += (MIST_ON_KANGASKHAN_VS_ALAKAZAM_REWARD
                                  * scale * ALAKAZAM_MIST_EMPHASIS)
                    mist_already += 1
            elif _kangaskhan_can_take_ko(cur_obs, me_index, opp_index, kang_energy_after):
                # This attachment turns straight into a knockout -- fine.
                target_reward += ENERGY_ON_KANGASKHAN_FOR_KO_REWARD
            elif not has_line:
                if could_develop:
                    # We were holding a way to put a body down first and
                    # attached to Kangaskhan instead, burning the turn's
                    # attachment on the wrong target.
                    target_reward -= ENERGY_ON_KANGASKHAN_MISORDERED_PENALTY
                else:
                    # Genuinely nothing to build. Not a mistake, just a bad
                    # board; the pressure lands on _tutor_reward instead.
                    target_reward -= ENERGY_ON_KANGASKHAN_NO_LINE_PENALTY
            elif line_available:
                target_reward -= ENERGY_ON_KANGASKHAN_PENALTY
            else:
                target_reward -= ENERGY_ON_KANGASKHAN_LINE_FULL_PENALTY

    return attach_reward, target_reward, zam_bonus


def _hero_cape_reward(prev_obs, cur_obs, me_index, matchup):
    """The Cape is a single copy and belongs on Crustle. The one board where
    Kangaskhan is a legal target is Alakazam, where it is the attacker.

    Recipients come from the +100 maxHp jump (see _tool_gains) rather than the
    Attach log, for the same reason the energy grading does.
    """
    total = 0.0
    vs_zam = matchup == MATCHUP_ALAKAZAM
    for target_id, _area in _tool_gains(prev_obs, cur_obs, me_index):
        if target_id == CRUSTLE:
            total += HERO_CAPE_ON_CRUSTLE_REWARD * (0.4 if vs_zam else 1.0)
        elif target_id == MEGA_KANGASKHAN_EX:
            total += (HERO_CAPE_ON_KANGASKHAN_VS_ALAKAZAM_REWARD if vs_zam
                      else -HERO_CAPE_ON_KANGASKHAN_PENALTY)
        else:
            total -= HERO_CAPE_WASTED_PENALTY
    return total


def _petrel_cape_reward(prev_obs, cur_obs, me_index, my_discards):
    """Petrel tutoring the single Hero's Cape out of the deck. Paid on the
    fetch; attaching it to Crustle is paid separately by _hero_cape_reward, so
    the full Petrel -> Cape -> Crustle chain collects both."""
    if TEAM_ROCKETS_PETREL not in my_discards:
        return 0.0
    gained = _newly_in_hand_ids(prev_obs, cur_obs, me_index)
    return PETREL_FETCHES_CAPE_REWARD if HEROS_CAPE in gained else 0.0


def _heal_reward(cur_obs, me_index, my_discards):
    """Ice Cream gets the threshold treatment; other healing is small and
    linear. Attribution is by "was a Jumbo Ice Cream discarded this step", so
    a step that both Ice Creams and heals another way lumps the total under
    the Ice Cream branch -- those rarely stack here."""
    healed = _heal_dealt(cur_obs, me_index)
    if healed <= 0:
        return 0.0
    if JUMBO_ICE_CREAM in my_discards:
        return ICE_CREAM_BASE + ICE_CREAM_PER_10HP * (healed / 10)
    return HEAL_REWARD_PER_10HP * (healed / 10)



def _hammer_reward(prev_obs, cur_obs, opp_index, my_discards):
    """Play, land, and target choice.

    Where the Energy came from is inferred from their Active's Energy count
    dropping, not from any target field on the play -- so it is skipped
    entirely when their Active changed identity this step, since a knockout or
    a switch also drops that count and would read as a hammer hit.
    """
    if CRUSHING_HAMMER not in my_discards:
        return 0.0
    reward = HAMMER_PLAY_REWARD
    if not any(_is_energy_card(cid)
               for cid in _newly_discarded_ids(prev_obs, cur_obs, opp_index)):
        return reward  # flipped tails
    reward += HAMMER_LAND_REWARD

    before = _active_pokemon(prev_obs, opp_index)
    if _active_identity(prev_obs, opp_index) != _active_identity(cur_obs, opp_index):
        return reward  # their Active moved; can't attribute the discard
    active_energy_before = len(_energies(before))
    if active_energy_before > _active_energy_count(cur_obs, opp_index):
        reward += HAMMER_ACTIVE_TARGET_REWARD
    elif active_energy_before > 0:
        reward -= HAMMER_BENCH_TARGET_PENALTY  # the Active was a legal target
    return reward


def _xerosic_reward(prev_obs, cur_obs, opp_index, my_discards):
    """Cuts them to 3, so the value is entirely in the excess.

    Scored off how many cards actually hit their discard, NOT off their
    pre-play hand size -- the opponent's hand is hidden, so the old hand-size
    version read 0 every time and this term never fired at all. If a count
    field for the hidden hand does resolve, it is used as a floor so the term
    still works on engines that expose one.
    """
    if XEROSICS_MACHINATIONS not in my_discards:
        return 0.0
    stripped = _opponent_cards_lost_from_hand(prev_obs, cur_obs, opp_index)
    known_excess = max(0, _hand_size(prev_obs, opp_index) - 3)
    return min(XEROSIC_MAX_REWARD, XEROSIC_PER_CARD_REWARD * max(stripped, known_excess))


def _hand_trimmer_reward(prev_obs, cur_obs, me_index, opp_index, my_discards):
    """Symmetric: both players cut to 5, opponent first. Good only when their
    hand is fat and ours is lean, so the self-discard is priced in rather than
    ignored."""
    if HAND_TRIMMER not in my_discards:
        return 0.0
    # Their side is measured from their discard for the same hidden-hand
    # reason as Xerosic; our side is a real card list so hand size is fine.
    their_loss = max(_opponent_cards_lost_from_hand(prev_obs, cur_obs, opp_index),
                     max(0, _hand_size(prev_obs, opp_index) - 5))
    # -1 for the Trimmer itself, which leaves our hand before the effect.
    our_loss = max(0, (_hand_size(prev_obs, me_index) - 1) - 5)
    return (min(TRIMMER_MAX_REWARD, TRIMMER_PER_CARD_REWARD * their_loss)
            - TRIMMER_SELF_DISCARD_PENALTY * our_loss)


def _boss_target_value(mon, matchup, can_punish):
    """How much we want THIS Pokemon standing in their Active Spot.

    Named threats first, then the generic "can Superb Scissors finish it"
    check, then -- only when nothing named or killable applies -- the fallback
    ordering the deck wants: weakest body available, then whatever is most
    expensive for them to retreat back out of.

    The 120 threshold ignores Weakness and Resistance, same approximation the
    starmie file made.
    """
    if mon is None:
        return 0.0
    cid = mon.get("id")
    hp = mon.get("hp")
    in_ko_range = isinstance(hp, (int, float)) and hp <= CRUSTLE_ATTACK_DAMAGE

    # A named threat's constant IS its priority, with one addition: being
    # ALSO inside KO range on top of being named means this Boss converts
    # straight into a knockout, which is worth more than just improving the
    # Active Spot (BOSS_NAMED_KO_BONUS). Letting the generic in-KO-range bonus
    # stack unconditionally used to reshuffle the intended order: an 80 HP
    # Makuhita picked up +0.03 that a 140 HP charged Hariyama could not, which
    # flipped the two -- the bonus here is per-branch on purpose.
    named = 0.0
    if cid in FROSLASS_LINE_IDS:
        # Froslass chips our Crustles through Rock Inn every Checkup -- the one
        # card that beats the gameplan by ignoring it, and it dies to 120.
        # Unconditional and matchup-agnostic on purpose: it turns up as a tech
        # in other lists too (Grimmsnarl among them), and it should always be
        # the pick over anything else on their bench.
        named = BOSS_FROSLASS_REWARD + (BOSS_NAMED_KO_BONUS if in_ko_range else 0.0)
    elif matchup == MATCHUP_BELLIBOLT and can_punish and in_ko_range:
        if cid == IONOS_VOLTORB:
            named = BOSS_VOLTORB_REWARD
        elif cid == IONOS_KILOWATTREL:
            named = BOSS_KILOWATTREL_REWARD
    elif matchup == MATCHUP_LUCARIO and can_punish:
        if cid == HARIYAMA:
            named = (BOSS_HARIYAMA_CHARGED_REWARD if _energies(mon)
                     else BOSS_HARIYAMA_UNCHARGED_REWARD)
            if in_ko_range:
                named += BOSS_NAMED_KO_BONUS
        elif cid == MAKUHITA:
            named = BOSS_MAKUHITA_REWARD
            if in_ko_range:
                named += BOSS_NAMED_KO_BONUS
    elif matchup == MATCHUP_MIRROR and can_punish and cid == CRUSTLE:
        # An under-charged mirror Crustle can't hit back at all if it's the
        # one standing Active -- free damage every turn it stays there.
        if len(_energies(mon)) < CRUSTLE_ATTACK_COST:
            named = BOSS_MIRROR_LOW_ENERGY_REWARD
    elif matchup == MATCHUP_ABOMASNOW and cid == KYOGRE_ME01:
        # Always the pick against Abomasnow, full HP or not -- prefer the
        # bench copy carrying the most Energy (the one closest to attacking).
        # UNVERIFIED id, see KYOGRE_ME01's definition.
        named = BOSS_ABOMASNOW_KYOGRE_REWARD + 0.001 * len(_energies(mon))
    if named > 0.0:
        return named

    if in_ko_range:
        value = BOSS_KO_RANGE_REWARD
        if _prize_value_by_id(cid) >= 2:
            value += BOSS_MULTI_PRIZE_REWARD
        return value

    # Nothing named and nothing killable: weakest body available, and prefer
    # one that is expensive to retreat back out of.
    value = 0.0
    max_hp = mon.get("maxHp")
    if isinstance(hp, (int, float)) and isinstance(max_hp, (int, float)) and max_hp > 0:
        value += BOSS_WEAK_TARGET_MAX_REWARD * max(0.0, 1.0 - (hp / 300.0))
    if _retreat_cost(mon) >= 2:
        value += BOSS_HIGH_RETREAT_REWARD
    return value


def _boss_reward(prev_obs, cur_obs, me_index, opp_index, my_discards, matchup):
    """Grade a Boss's Orders by how much it IMPROVED their Active Spot.

    Scoring the new target in isolation is what let the bot drag a Mega Lucario
    ex up while a Hariyama was already Active: Lucario is a heavy retreater, so
    the fallback high-retreat bonus paid out even though the swap traded a
    killable threat for one Crustle can neither damage nor be damaged by.
    Comparing against what was already standing there fixes that class of
    mistake generally, not just for Lucario.

    Reads the post-effect board only, so there is no credit assignment to
    whichever later attack actually converts the knockout.
    """
    if BOSSS_ORDERS not in my_discards:
        return 0.0
    target = _active_pokemon(cur_obs, opp_index)
    if not target:
        return 0.0

    can_punish = _ready_crustle_anywhere(prev_obs, me_index)
    before = _active_pokemon(prev_obs, opp_index)

    # Giving up a FREE knockout. If whatever was already Active was inside
    # Superb Scissors range and we could already punish it, the only reason to
    # spend Boss's Orders is to drag up something ALSO in range -- a bigger or
    # multi-prize kill. Trading that guaranteed KO for a fresh full-HP body,
    # named threat or not, undoes real progress instead of making any -- this
    # is the case that let the bot boss up a full-HP target while the current
    # Active (sometimes the very same species) was already sitting in KO
    # range.
    before_hp = before.get("hp") if before is not None else None
    before_in_ko_range = (can_punish and isinstance(before_hp, (int, float))
                           and before_hp <= CRUSTLE_ATTACK_DAMAGE)
    target_hp = target.get("hp")
    target_in_ko_range = isinstance(target_hp, (int, float)) and target_hp <= CRUSTLE_ATTACK_DAMAGE
    if before_in_ko_range and not target_in_ko_range:
        return -BOSS_DOWNGRADE_PENALTY

    new_value = _boss_target_value(target, matchup, can_punish)
    old_value = _boss_target_value(before, matchup, can_punish)

    # Nothing moved -- Boss was played on whoever was already Active.
    if before is not None and target.get("id") == before.get("id") and old_value == new_value:
        return -BOSS_WASTED_PENALTY

    gain = new_value - old_value
    if gain <= 0.0:
        return -BOSS_DOWNGRADE_PENALTY
    if new_value <= 0.0:
        return -BOSS_WASTED_PENALTY
    return gain


def _switch_reward(prev_obs, cur_obs, me_index, my_discards, matchup):
    """Only pays when the Switch actually LANDS a built Crustle in the Active
    Spot. The previous version rewarded any swap out of a "stuck" Active,
    which the agent was cashing on turn 1 by Switching Kangaskhan away for
    nothing -- grading the post-switch board removes that entirely.

    A Switch that does NOT land a built Crustle is now charged rather than
    left neutral -- the old 0.0 was what let Switch get burned at random
    (including on a Kangaskhan Active that was doing its job) with nothing
    pulling the agent back. In the mirror the whole card is worth far more
    (see SWITCH_KANGASKHAN_TO_CRUSTLE_MIRROR_REWARD's definition), so both the
    miss-penalty and the specific Kangaskhan-out play are scaled up there.
    """
    if SWITCH not in my_discards:
        return 0.0
    prev_active = _active_pokemon(prev_obs, me_index)
    mon = _active_pokemon(cur_obs, me_index)
    energy = len(_energies(mon)) if mon else 0
    landed_ready_crustle = bool(mon and mon.get("id") == CRUSTLE and energy >= CRUSTLE_ATTACK_COST)

    if matchup == MATCHUP_MIRROR:
        if (landed_ready_crustle and prev_active is not None
                and prev_active.get("id") == MEGA_KANGASKHAN_EX):
            fill = min(1.0, energy / CRUSTLE_MAX_USEFUL_ENERGY)
            return SWITCH_KANGASKHAN_TO_CRUSTLE_MIRROR_REWARD * (0.6 + 0.4 * fill)
        if not landed_ready_crustle:
            return -SWITCH_WASTED_MIRROR_PENALTY

    if not landed_ready_crustle:
        return -SWITCH_WASTED_PENALTY

    # Prefer the fuller Crustle when more than one is switch-in eligible: half
    # the reward right at attack cost, all of it at the HP-chasing ceiling.
    fill = min(1.0, energy / CRUSTLE_MAX_USEFUL_ENERGY)
    return SWITCH_TO_BUILT_CRUSTLE_REWARD * (0.5 + 0.5 * fill)


def _pokegear_reward(prev_obs, cur_obs, me_index, opp_index, my_discards, matchup):
    if POKEGEAR_3 not in my_discards:
        return 0.0
    wanted = _wanted_supporter_ids(prev_obs, me_index, opp_index, matchup)
    supporters = [cid for cid in _newly_in_hand_ids(prev_obs, cur_obs, me_index)
                  if _is_supporter_card(cid)]
    if not supporters:
        return 0.0
    return POKEGEAR_WANTED_REWARD if any(c in wanted for c in supporters) else POKEGEAR_ANY_REWARD


def _lillie_reward(prev_obs, me_index, my_discards):
    """Two regimes, because Lillie means opposite things depending on the deck.

    Normally it shuffles the hand away for 6, so the cost is every card it
    throws away and it is best on an empty hand.

    Under deck-out pressure that flips: the shuffle-back is the only way this
    list puts cards BACK into the deck, and the net change is (hand - 6). So
    once the deck is thin we want it held for a fat hand and charged for
    burning it on a lean one. The counted hand is capped at
    LILLIE_SWEET_SPOT_HAND so there is no gradient encouraging the agent to
    keep stockpiling -- cards held past that point are just food for an
    opposing Xerosic's, which cuts us to 3 no matter how many we had.
    """
    if LILLIES_DETERMINATION not in my_discards:
        return 0.0
    hand_before = max(0, _hand_size(prev_obs, me_index) - 1)  # Lillie itself has left

    if _deck_remaining(prev_obs, me_index) <= LILLIE_DECKOUT_THRESHOLD:
        counted = min(hand_before, LILLIE_SWEET_SPOT_HAND - 1)
        net_deck_gain = counted - 6
        if net_deck_gain <= 0:
            return -LILLIE_DECKOUT_EARLY_PENALTY
        return min(LILLIE_DECKOUT_MAX_REWARD, LILLIE_DECK_GAIN_PER_CARD * net_deck_gain)

    return max(LILLIE_MAX_PENALTY, LILLIE_BASE_REWARD - LILLIE_PER_CARD_DISCARDED * hand_before)


def _hilda_reward(prev_obs, cur_obs, me_index, my_discards, matchup):
    """Hilda actually fetching a Crustle, a Growing Grass Energy, or a Mist
    Energy -- the three things this deck actually wants out of it, priced
    separately so the term stops under-rewarding the two-thirds of Hilda's
    value (Crustle, grass) that used to score nothing at all.

    Paid on the fetch; attaching an Energy is paid separately by the relevant
    energy_target / mist_vs_alakazam term, so the full Hilda -> card ->
    board chain collects both. Against Alakazam the Mist payout decays with
    every turn that has already passed and shrinks as we accumulate copies
    (smoothed continuously rather than in hard steps -- see the urgency
    line below), so the gradient points at "find all four early" rather than
    "find one eventually".
    """
    if HILDA not in my_discards:
        return 0.0
    gained = _newly_in_hand_ids(prev_obs, cur_obs, me_index)
    if not gained:
        return 0.0

    reward = 0.0
    if CRUSTLE in gained:
        reward += HILDA_CRUSTLE_REWARD
    reward += HILDA_GRASS_ENERGY_REWARD * sum(1 for cid in gained if cid == GROW_GRASS_ENERGY)

    mist_found = sum(1 for cid in gained if cid == MIST_ENERGY)
    if mist_found <= 0:
        return reward
    if matchup != MATCHUP_ALAKAZAM:
        return reward + HILDA_MIST_REWARD * mist_found

    per_copy = max(HILDA_MIST_MIN_REWARD,
                   HILDA_MIST_VS_ALAKAZAM_REWARD
                   - HILDA_MIST_DECAY_PER_TURN * _turns_taken.get(me_index, 0))
    # The copies we don't have yet are the urgent ones. Continuous in
    # still_missing (0.5 at 0 missing up to 1.0 at all 4 missing) rather than
    # the old hard 0.5/0.75/1.0 steps, which were a real source of the jagged,
    # noisy-looking reward/hilda_mist series -- two otherwise-similar fetches
    # a turn apart could land on opposite sides of a step and look like noise
    # rather than the smooth "find them early" signal this is meant to be.
    still_missing = max(0, 4 - _mist_secured(prev_obs, me_index))
    urgency = 0.5 + 0.125 * still_missing
    return reward + per_copy * urgency * mist_found * ALAKAZAM_MIST_EMPHASIS


def _tutor_reward(prev_obs, me_index, my_discards):
    """Ultra Ball / Hilda into a stranded Dwebble, Petrel or Poffin when the
    board is short of the two Crustle the deck wants, or when we need to dig
    for a heal."""
    reward = 0.0
    stranded = _stranded_dwebble(prev_obs, me_index)
    line_count = _crustle_line_count(prev_obs, me_index)
    short_of_pair = line_count < 2
    # An empty board is the urgent case: with no Dwebble or Crustle out there
    # is nothing to build, Kangaskhan soaks the energy by default, and the
    # whole gameplan is stalled until a body shows up.
    payout = TUTOR_NO_LINE_REWARD if line_count == 0 else TUTOR_ON_NEED_REWARD
    hand = set(_hand_ids(prev_obs, me_index))
    # Nothing left in hand that can put a body down except Petrel digging for
    # a Poffin -- see PETREL_LAST_RESORT_REWARD.
    no_dwebble_access = (line_count == 0
                          and not (hand & {DWEBBLE, BUDDY_BUDDY_POFFIN, ULTRA_BALL}))
    for cid in my_discards:
        if cid == HILDA and (stranded or short_of_pair):
            # A Pokemon AND an Energy off one card beats Ultra Ball's Pokemon-
            # only for the same slot, so it gets a bit extra on top.
            reward += payout + HILDA_TUTOR_BONUS
        elif cid == ULTRA_BALL and (stranded or short_of_pair):
            reward += payout
        elif cid == TEAM_ROCKETS_PETREL:
            if no_dwebble_access:
                reward += payout + PETREL_LAST_RESORT_REWARD
            elif short_of_pair or _wants_ice_cream(prev_obs, me_index):
                reward += payout
        elif cid == BUDDY_BUDDY_POFFIN:
            if short_of_pair or _wants_ice_cream(prev_obs, me_index):
                reward += payout
    return reward


def _promotion_reward(prev_obs, cur_obs, me_index, opp_took, matchup):
    """Grade the replacement chosen after our Active was knocked out.

    Ranking, in order: anything holding Mist Energy (it walks into the next
    attack immune to its effects), then the biggest Crustle -- or Kangaskhan
    in the Alakazam matchup, where it is the attacker. Only fires on steps
    where the opponent actually took a Prize, which is what tells us the
    Active change was a knockout rather than a Switch or a retreat.
    """
    if opp_took <= 0:
        return 0.0
    promoted = _active_pokemon(cur_obs, me_index)
    candidates = _bench_pokemon(prev_obs, me_index)
    if not promoted or not candidates:
        return 0.0

    def rank(mon):
        if _has_mist(mon):
            return 1000 + len(_energies(mon))
        if matchup == MATCHUP_ALAKAZAM and mon.get("id") == MEGA_KANGASKHAN_EX:
            return 500
        if mon.get("id") == CRUSTLE:
            return 100 + len(_energies(mon)) * 10
        return 0

    best = max(rank(m) for m in candidates)
    return PROMOTION_BEST_REWARD if rank(promoted) >= best else -PROMOTION_WRONG_PENALTY


def _setup_tempo_reward(cur_obs, me_index):
    """One-shot the first time a Crustle reaches attack cost, decayed by how
    many of our turns have already ended. Rewards reaching the plan, not
    maintaining it -- a rebuilt Crustle doesn't pay again."""
    if _setup_scored.get(me_index):
        return 0.0
    if _best_crustle_energy(cur_obs, me_index) < CRUSTLE_ATTACK_COST:
        return 0.0
    _setup_scored[me_index] = True
    return max(0.0, SETUP_TEMPO_BASE - SETUP_TEMPO_DECAY_PER_TURN * _turns_taken.get(me_index, 0))


def _crustle_pair_reward(cur_obs, me_index):
    """One-shot for getting TWO Crustle onto the board. The single wall keeps
    dying to prize trades and non-ex attackers; the deck wants a spare."""
    if _pair_scored.get(me_index):
        return 0.0
    if _count_in_play(cur_obs, me_index, CRUSTLE) < 2:
        return 0.0
    _pair_scored[me_index] = True
    return CRUSTLE_PAIR_REWARD


def _wall_matchup_reward(prev_obs, cur_obs, me_index, opp_index):
    """Crustle attacking into a Pokemon ex: Rock Inn means that defender can't
    damage it back, so the hit is free. Defender read from prev_obs, since the
    KO may already have cleared the Active Spot by cur_obs."""
    if _attacker_id(cur_obs, me_index) != CRUSTLE:
        return 0.0
    defender = _active_pokemon(prev_obs, opp_index)
    if not defender:
        return 0.0
    return CRUSTLE_WALL_ATTACK_REWARD if _is_ex_pokemon(defender.get("id")) else 0.0


def _kangaskhan_attack_term(prev_obs, cur_obs, me_index, matchup):
    """Kangaskhan should be drawing, not swinging -- it is a 3-prize body and
    every attack exposes it while the bench Crustle goes unbuilt. Flat charge
    plus more when a built Crustle was available, and the whole thing inverts
    against Alakazam where Kangaskhan IS the attacker."""
    if _attacker_id(cur_obs, me_index) != MEGA_KANGASKHAN_EX:
        return 0.0
    if matchup == MATCHUP_ALAKAZAM:
        return KANGASKHAN_ATTACK_VS_ALAKAZAM_REWARD
    penalty = KANGASKHAN_ATTACK_PENALTY
    if _ready_crustle_anywhere(prev_obs, me_index):
        penalty += KANGASKHAN_ATTACK_WITH_CRUSTLE_READY_PENALTY
    return -penalty


def _active_identity(obs_dict, player_index):
    """Serial if the observation carries one, else the card id -- two copies of
    the same Pokemon across a step would otherwise look like no change."""
    mon = _active_pokemon(obs_dict, player_index)
    if not mon:
        return None
    return mon.get("serial", mon.get("id"))


def _retreat_reward(prev_obs, cur_obs, me_index, my_discards, opp_took, matchup):
    """Grade a retreat: charge for Energy dumped paying the cost, EXCEPT the
    one retreat this deck actually wants -- Kangaskhan can't (or shouldn't be
    trusted to) attack and a ready Crustle is waiting on the bench, which is
    the transition from "Kangaskhan draws" to "Crustle attacks". That specific
    retreat pays a flat reward instead of the energy charge, which is what the
    old unconditional version was missing (it charged every retreat the same
    way regardless of whether it was the correct play).

    INFERRED: there is no Retreat log type in the set this file reads, so a
    retreat is inferred from "our Active changed, our own Energy hit the
    discard, no Switch was played, and the opponent took no Prize". The Prize
    guard is what separates a retreat from a knockout. A Boss played by the
    opponent moves our Active without discarding our Energy, so it won't fire.
    """
    if opp_took > 0 or SWITCH in my_discards:
        return 0.0
    if _active_identity(prev_obs, me_index) == _active_identity(cur_obs, me_index):
        return 0.0

    prev_active = _active_pokemon(prev_obs, me_index)
    new_active = _active_pokemon(cur_obs, me_index)
    landed_ready_crustle = (new_active is not None and new_active.get("id") == CRUSTLE
                             and len(_energies(new_active)) >= CRUSTLE_ATTACK_COST)

    if (matchup != MATCHUP_ALAKAZAM and landed_ready_crustle
            and prev_active is not None and prev_active.get("id") == MEGA_KANGASKHAN_EX):
        unable_to_attack = (len(_energies(prev_active)) < KANGASKHAN_ATTACK_COST
                             or _has_special_condition(prev_active, "ASLEEP")
                             or _has_special_condition(prev_active, "PARALY"))
        hp, max_hp = prev_active.get("hp"), prev_active.get("maxHp")
        damaged = (isinstance(hp, (int, float)) and isinstance(max_hp, (int, float))
                   and hp < max_hp)
        if (unable_to_attack or damaged) and _ready_crustle_on_bench(prev_obs, me_index):
            return RETREAT_TO_READY_CRUSTLE_REWARD

    return -RETREAT_ENERGY_PENALTY * sum(1 for cid in my_discards if _is_energy_card(cid))


def _deck_out_penalty(prev_obs, cur_obs, me_index):
    """Charged once on each threshold crossing, not per step while thin. With
    the draw reward removed, this is the only thing pushing back on drawing
    ourselves out -- which a grind deck is the archetype most likely to do."""
    prev_deck = _deck_remaining(prev_obs, me_index)
    cur_deck = _deck_remaining(cur_obs, me_index)
    penalty = 0.0
    if prev_deck > DECK_LOW_THRESHOLD >= cur_deck:
        penalty += DECK_LOW_PENALTY
    if prev_deck > DECK_CRITICAL_THRESHOLD >= cur_deck:
        penalty += DECK_CRITICAL_PENALTY
    return penalty


def _board_shape_penalty(prev_obs, cur_obs, me_index):
    """Charged on the transition that pushes the board past the caps, NOT per
    step while over. A per-step version across a several-hundred-step episode
    is an enormous hidden penalty against a +/-1.0 terminal."""
    penalty = 0.0
    prev_k = _count_in_play(prev_obs, me_index, MEGA_KANGASKHAN_EX)
    cur_k = _count_in_play(cur_obs, me_index, MEGA_KANGASKHAN_EX)
    if cur_k > prev_k:
        over = max(0, cur_k - MAX_KANGASKHAN_IN_PLAY) - max(0, prev_k - MAX_KANGASKHAN_IN_PLAY)
        penalty += EXTRA_KANGASKHAN_PENALTY * max(0, over)

    prev_line = _crustle_line_count(prev_obs, me_index)
    cur_line = _crustle_line_count(cur_obs, me_index)
    if cur_line > prev_line:
        over = (max(0, cur_line - MAX_CRUSTLE_LINE_IN_PLAY)
                - max(0, prev_line - MAX_CRUSTLE_LINE_IN_PLAY))
        penalty += EXTRA_CRUSTLE_LINE_PENALTY * max(0, over)
    return penalty


def _kangaskhan_bench_penalty(prev_obs, cur_obs, me_index, matchup):
    """Charged ONLY in the Mega Lucario matchup, and only when a new Kangaskhan
    lands on the BENCH.

    Outside Lucario this returns 0 by design: MAX_KANGASKHAN_IN_PLAY already
    holds the board to one, and layering a second penalty on top of that cap
    was pushing the trained bot away from a board state that is actually fine.

    The opening Kangaskhan is exempt even against Lucario -- it is the correct
    lead. Because a Basic played from hand can only go to the Bench, "new copy
    on the Bench" is exactly "played another one after setup", so no turn
    counter is needed. Moving an existing Kangaskhan out of the Active Spot
    doesn't count either: the total in-play count has to rise as well, so
    retreating it to safety is free. Nothing here depends on CARD_DB.
    """
    if matchup != MATCHUP_LUCARIO:
        return 0.0
    added_total = (_count_in_play(cur_obs, me_index, MEGA_KANGASKHAN_EX)
                   - _count_in_play(prev_obs, me_index, MEGA_KANGASKHAN_EX))
    if added_total <= 0:
        return 0.0

    def bench_count(o):
        return sum(1 for m in _bench_pokemon(o, me_index)
                   if m.get("id") == MEGA_KANGASKHAN_EX)

    landed_on_bench = min(added_total, max(0, bench_count(cur_obs) - bench_count(prev_obs)))
    return KANGASKHAN_BENCHED_VS_LUCARIO_PENALTY * landed_on_bench


def _stadium_reward(prev_obs, cur_obs, me_index, opp_index):
    """Play, plus a bump bonus, plus extra for Battle Cage specifically
    against decks that place damage counters -- it stops Freezing Shroud and
    Powerful Hand from touching the Bench where the Crustles are built."""
    prev_stadium = _stadium_identity(prev_obs)
    cur_stadium = _stadium_identity(cur_obs)
    if cur_stadium is None or cur_stadium[1] != me_index or cur_stadium == prev_stadium:
        return 0.0
    reward = STADIUM_PLAY_REWARD
    if prev_stadium is not None and prev_stadium[1] == opp_index:
        reward += STADIUM_BUMP_REWARD
    if cur_stadium[0] == BATTLE_CAGE and _opponent_plays_counters(prev_obs, opp_index):
        reward += BATTLE_CAGE_VS_COUNTER_DECK_REWARD
    if cur_stadium[0] == FESTIVAL_GROUNDS:
        # Only pays if it actually frees something: the card works on Pokemon
        # that have Energy attached, so a bare Active gets nothing from it.
        active = _active_pokemon(prev_obs, me_index)
        if active and _energies(active):
            if _has_special_condition(active, "CONFUS"):
                reward += FESTIVAL_GROUNDS_CLEARS_CONFUSION_REWARD
            elif _has_special_condition(active):
                reward += FESTIVAL_GROUNDS_CLEARS_CONDITION_REWARD
    return reward


def _supporter_played_count(prev_obs, cur_obs, player_index):
    """Flat per-Supporter reward for playing ANY Supporter this turn --
    EXCEPT Boss's Orders, which already has its own discerning reward
    (_boss_reward). Stacking the flat bonus on top of a bad Boss play was
    softening the penalty for dragging up a bad target, which is exactly the
    "just Boss whatever, it still pays a little" behavior this exclusion
    exists to stop encouraging."""
    return sum(1 for cid in _newly_discarded_ids(prev_obs, cur_obs, player_index)
               if _is_supporter_card(cid) and cid != BOSSS_ORDERS)


# Fixed key set so the logger writes a COMPLETE series every rollout -- a term
# whose schema assumptions are wrong then shows up as a flat 0.0 line instead
# of silently never appearing at all.
REWARD_TERMS = (
    "terminal",
    "prize_mine",
    "prize_opp",
    "damage",
    "energy_attach",
    "energy_target",
    "mist_vs_alakazam",
    "evolve",
    "crustle_evolve",
    "crustle_pair",
    "hero_cape",
    "petrel_cape",
    "heal",
    "hammer",
    "xerosic",
    "hand_trimmer",
    "boss",
    "switch_to_crustle",
    "pokegear",
    "lillie",
    "tutor",
    "hilda_mist",
    "promotion",
    "stadium",
    "supporter",
    "board_shape",
    "kangaskhan_bench",
    "kangaskhan_attack",
    "wall_matchup",
    "setup_tempo",
    "retreat",
    "deck_out",
    "run_errand",
    "no_attack",
    "lead",
)


def reward_terms(prev_obs, cur_obs, done, result, me_index):
    """Named breakdown of compute_reward's terms, keyed for per-component
    logging (reward/<key> series).

    A terminal step returns a single {"terminal": +-1.0} entry -- nothing to
    shape once the game is over. NOT side-effect free: it advances the turn,
    lead, setup and matchup state, so call it once per step and sum the
    result rather than calling it and compute_reward both.
    """
    if done:
        return {"terminal": 1.0 if result == me_index else -1.0}

    opp_index = 1 - me_index
    matchup = _update_matchup(cur_obs, me_index, opp_index)
    my_discards = _newly_discarded_ids(prev_obs, cur_obs, me_index)
    # Tracked before the energy terms are graded: whether the turn's Supporter
    # is already spent decides if Hilda/Petrel counted as a development option.
    if any(_is_supporter_card(cid) for cid in my_discards):
        _turn_supporter_used[me_index] = True

    my_took = max(0, _prizes_remaining(prev_obs, me_index) - _prizes_remaining(cur_obs, me_index))
    opp_took = max(0, _prizes_remaining(prev_obs, opp_index) - _prizes_remaining(cur_obs, opp_index))
    my_prize_reward = PRIZE_REWARD * my_took
    if my_took >= 2:
        my_prize_reward *= MULTI_PRIZE_MULTIPLIER

    energy_attach, energy_target, zam_bonus = _energy_terms(
        prev_obs, cur_obs, me_index, opp_index, matchup)

    crustle_evolved = max(0, _count_in_play(cur_obs, me_index, CRUSTLE)
                          - _count_in_play(prev_obs, me_index, CRUSTLE))

    # Settles the per-turn accumulators and advances the turn counter; must be
    # called exactly once per step.
    run_errand, no_attack = _turn_end_penalties(prev_obs, cur_obs, me_index, matchup)

    terms = {
        "prize_mine": my_prize_reward,
        "prize_opp": -PRIZE_REWARD * opp_took,
        "damage": DAMAGE_REWARD_PER_100 * (_damage_dealt(cur_obs, opp_index) / 100),
        "energy_attach": energy_attach,
        "energy_target": energy_target,
        "mist_vs_alakazam": zam_bonus,
        "evolve": EVOLVE_REWARD * _evolve_count(cur_obs, me_index),
        # Counted off the board diff rather than the Evolve log's cardId /
        # cardIdTarget pairing, whose direction isn't confirmed.
        "crustle_evolve": CRUSTLE_EVOLVE_REWARD * crustle_evolved,
        "crustle_pair": _crustle_pair_reward(cur_obs, me_index),
        "hero_cape": _hero_cape_reward(prev_obs, cur_obs, me_index, matchup),
        "petrel_cape": _petrel_cape_reward(prev_obs, cur_obs, me_index, my_discards),
        "heal": _heal_reward(cur_obs, me_index, my_discards),
        "hammer": _hammer_reward(prev_obs, cur_obs, opp_index, my_discards),
        "xerosic": _xerosic_reward(prev_obs, cur_obs, opp_index, my_discards),
        "hand_trimmer": _hand_trimmer_reward(prev_obs, cur_obs, me_index, opp_index, my_discards),
        "boss": _boss_reward(prev_obs, cur_obs, me_index, opp_index, my_discards, matchup),
        "switch_to_crustle": _switch_reward(prev_obs, cur_obs, me_index, my_discards, matchup),
        "pokegear": _pokegear_reward(prev_obs, cur_obs, me_index, opp_index, my_discards, matchup),
        "lillie": _lillie_reward(prev_obs, me_index, my_discards),
        "tutor": _tutor_reward(prev_obs, me_index, my_discards),
        "hilda_mist": _hilda_reward(
            prev_obs, cur_obs, me_index, my_discards, matchup),
        "promotion": _promotion_reward(prev_obs, cur_obs, me_index, opp_took, matchup),
        "stadium": _stadium_reward(prev_obs, cur_obs, me_index, opp_index),
        "supporter": SUPPORTER_PLAY_REWARD * _supporter_played_count(prev_obs, cur_obs, me_index),
        "board_shape": -_board_shape_penalty(prev_obs, cur_obs, me_index),
        "kangaskhan_bench": -_kangaskhan_bench_penalty(
            prev_obs, cur_obs, me_index, matchup),
        "kangaskhan_attack": _kangaskhan_attack_term(prev_obs, cur_obs, me_index, matchup),
        "wall_matchup": _wall_matchup_reward(prev_obs, cur_obs, me_index, opp_index),
        "setup_tempo": _setup_tempo_reward(cur_obs, me_index),
        "retreat": _retreat_reward(prev_obs, cur_obs, me_index, my_discards, opp_took, matchup),
        "deck_out": -_deck_out_penalty(prev_obs, cur_obs, me_index),
        "run_errand": -run_errand,
        "no_attack": -no_attack,
        "lead": _lead_reward(prev_obs, cur_obs, me_index),
    }

    if SHAPING_SCALE != 1.0:
        terms = {k: v * SHAPING_SCALE for k, v in terms.items()}
    return terms


def compute_reward(prev_obs, cur_obs, done, result, me_index):
    """Total reward from the acting player's perspective -- the sum of
    reward_terms() (see it for the term list and the arguments).

    Single-value entry point for callers that don't need the breakdown.
    Callers that want both should call reward_terms() once and sum it, since
    reward_terms() is not side-effect free.
    """
    return sum(reward_terms(prev_obs, cur_obs, done, result, me_index).values())


def current_matchup(me_index=0):
    """Which archetype tag the shaping is currently applying, for debugging a
    run where the Alakazam inversions look like they fired at the wrong time."""
    return _matchup.get(me_index, MATCHUP_GENERIC)
