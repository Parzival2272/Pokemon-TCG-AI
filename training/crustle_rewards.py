"""Reward shaping for the RL learner -- Mega Kangaskhan ex / Crustle list.

Same contract as the Mega Starmie ex version of this file: rewards are always
from the perspective of the player who took the step's action ("me", an
absolute player index), the base signal is the sparse terminal +1 win / -1
loss, and on non-terminal steps we add dense shaping. `compute_reward` is the
single-value entry point; `reward_terms` returns the named breakdown for
per-component TensorBoard logging.

DECK PLAN THE SHAPING ENCODES
-----------------------------
Mega Kangaskhan ex (300 HP, {C}, Basic) sits in the Active Spot drawing 2 a
turn with Run Errand while Crustle gets built on the bench. Crustle (150 HP,
Stage 1) has Mysterious Rock Inn -- it takes *zero* damage from attacks by
opposing Pokemon ex -- and attacks for 120 with Superb Scissors at {G}{C}{C},
ignoring effects on the defender. Growing Grass Energy adds +20 HP each and
Hero's Cape another +100, so a fully-dressed Crustle is a 250-330 HP wall that
an ex-based deck literally cannot damage. Once Crustle has 3 energy we Switch
it in (retreat cost is 3, so Switch/Petrel-for-Switch is the real swap) and
grind: Jumbo Ice Cream heals 80, Crushing Hammer/Xerosic/Eri strip the
opponent's resources, Boss's Orders drags out whatever the wall can punish.

The shaping therefore pushes: energy onto Crustle (never Kangaskhan), Mist
Energy first, evolve the line, cape the wall, heal at full value, deny
resources, and keep exactly 1 Kangaskhan + 2 line members on board.

Two named threats the shaping treats specially:
  * Froslass TWM 53 (and Snorunt, its Basic): Freezing Shroud puts a damage
    counter on every Pokemon with an Ability each Checkup. Crustle has an
    Ability, so Froslass chips through the ex wall -- it's the one card that
    beats the gameplan by ignoring it. At 90 HP (Snorunt 60) Superb Scissors
    one-shots it, so Boss's Orders onto it is heavily rewarded.
  * Any Fighting Pokemon: Kangaskhan is {C} with Fighting x2 weakness and
    gives up 3 prizes. Putting a fresh one down into a Fighting board is a
    large penalty.

WHAT I CHANGED VS THE STARMIE FILE (read before porting anything back)
---------------------------------------------------------------------
1. `_energies()` reads "energies" first, falling back to "energyCards".
   Both keys exist on a live Pokemon: `energies` is list[EnergyType] (what an
   attack cost is paid with) and `energyCards` is list[Card]. Every caller
   here is a cost threshold, so `energies` is the right one; the starmie
   file's hardcoded "energyCards" happens to give the same count for basic
   Energy but diverges on anything providing 0 or 2+ units.
2. The bench-size penalty is TRANSITION-based, not state-based. The starmie
   file charges BENCH_OVER_4_PENALTY on *every decision step* while the board
   is too wide, which for a several-hundred-step episode is an enormous
   hidden penalty relative to a +/-1.0 terminal. Here the board-shape penalty
   fires only on the step where the count crosses the cap.
3. Every shaping term is multiplied by SHAPING_SCALE so the whole block can be
   dialled down with one constant if the measured per-episode totals come in
   hot. Measure first (the starmie file's post-mortem comments are the
   cautionary tale), then tune.

VERIFICATION STATUS: every card ID below was checked against CARD_DB by name,
and the observation schema was confirmed by instrumenting real games (see the
post-mortem below). `cardIdTarget` on Attach entries -- previously only
assumed -- does carry the receiving Pokemon's card ID, so the energy_target
and hero_cape terms are sound.

Two properties of `logs` that the terms here depend on, both confirmed:
  * It is a PER-VIEWER window: the engine reports each event exactly once to
    each player, covering everything since *that* player's previous decision.
    So a learner step that spans the opponent's whole turn sees that turn's
    events once, and nothing is double counted -- provided the observation
    being read belongs to the same viewer as `me_index`. That holds for every
    CabtEnv path that has an opponent agent (the learner is a fixed side); it
    does NOT hold for CabtEnv's pure-self-play fallback, where cur_obs may be
    rendered for the other player.
  * TURN_END carries the playerIndex of the player whose turn ended, which is
    what _turn_end_penalties keys off.

POST-MORTEM: MaskablePPO_18 (why 20 of the 31 series were flat 0.0)
-------------------------------------------------------------------
Run 18 logged exactly the terms this file shares with training/rewards.py and
zeroed every crustle-specific one. Cause: only callbacks.py had been switched
to this module -- CabtEnv still imported training.rewards and still played the
starmie deck.csv, so RewardTermCallback was writing this file's key list
against another file's numbers, and the keys with no counterpart came out 0.
Four genuine bugs in here were hidden behind that and are now fixed:
  * `deck` is not a PlayerState field (it is `deckCount`)  -> deck_out
  * `hand` is None for the opponent (use `handCount`)      -> xerosic
  * CardType has no ENERGY member, only BASIC_/SPECIAL_    -> hammer, retreat
  * CardData.energyType is a raw int, not an enum          -> kangaskhan_fighting
plus turn hand-off being read off a yourIndex flip that the env never shows
the learner (-> no_attack, run_errand) and nobody calling reset_game_state()
between episodes (-> lead, setup_tempo latched after one game per worker).

Terms that log a flat 0.0 for a whole run are broken, not inactive -- that is
what REWARD_TERMS exists to make visible.
"""

from collections import Counter


# ── Global dial ───────────────────────────────────────────────────────────
# Multiplies every non-terminal term. Rough budget at 1.0, for a game played
# roughly to plan: prizes +/-0.6, damage ~0.12, draw ~0.06, setup/tempo
# one-shots ~0.13, everything else ~0.4 combined -- i.e. shaping can exceed
# the magnitude of the terminal reward on a long game. If training shows the
# agent farming shaping instead of winning, drop this to 0.5 before touching
# individual constants. The repeatable-per-turn terms are the ones to watch
# in the logs (draw, damage, energy_attach, run_errand, wall_matchup); the
# one-shots (lead, setup_tempo) and the once-per-copy card terms are bounded
# by the decklist and can't run away.
SHAPING_SCALE = 1.0

# ── Core (inherited) terms ────────────────────────────────────────────────
PRIZE_REWARD = 0.1
MULTI_PRIZE_MULTIPLIER = 1.25
DAMAGE_REWARD_PER_100 = 0.01
DRAW_REWARD_PER_CARD = 0.001
SUPPORTER_PLAY_REWARD = 0.005
NO_ATTACK_PENALTY = 0.05

# ── Energy attachment ─────────────────────────────────────────────────────
# Base for any energy attached (never wasting the once-per-turn attachment),
# plus a per-card bonus. Mist Energy is the highest-value attach in the deck:
# it blanks attack *effects* on the holder, which is the hole in Crustle's
# ex-damage wall (effect damage, status, forced switching all get through
# Mysterious Rock Inn). Growing Grass is next (+20 HP each on a Grass
# Pokemon), then Spiky (2 counters back on anything that hits our Active).
ENERGY_ATTACH_REWARD = 0.004
ENERGY_BONUS_BY_ID = {}  # populated after the IDs below

# Where the energy landed. Crustle/Dwebble is the plan; Kangaskhan is a
# 3-prize liability that only needs energy in an emergency, so energy on it
# is charged *only* when a Crustle line member was available to take it.
ENERGY_ON_CRUSTLE_LINE_REWARD = 0.008
ENERGY_ON_KANGASKHAN_PENALTY = 0.010

# ── Development ───────────────────────────────────────────────────────────
EVOLVE_REWARD = 0.005          # any evolution
CRUSTLE_EVOLVE_REWARD = 0.025  # specifically Dwebble -> Crustle

# Hero's Cape (+100 HP) attached to a Crustle. On anything else it is close
# to a dead card in this list.
HERO_CAPE_ON_CRUSTLE_REWARD = 0.04
HERO_CAPE_WASTED_PENALTY = 0.02

# ── Healing ───────────────────────────────────────────────────────────────
# Jumbo Ice Cream heals 80 from an Active with 3+ energy. Firing it to heal
# 20 burns one of 4 copies for a quarter of its value, so the term is a fixed
# charge plus a per-10-HP bonus: break-even around 35 HP, best at the full 80.
ICE_CREAM_BASE = -0.020
ICE_CREAM_PER_10HP = 0.006

# Any other healing we cause (Bianca's Devotion full-heals a Pokemon at <=30
# HP remaining, Community Center's 10-to-each). Linear and small; Bianca is
# self-gating on the card so it doesn't need a threshold here.
HEAL_REWARD_PER_10HP = 0.003

# ── Resource denial ───────────────────────────────────────────────────────
HAMMER_PLAY_REWARD = 0.004     # played at all (the flip can miss)
HAMMER_LAND_REWARD = 0.020     # opponent actually lost an Energy

# Xerosic's Machinations forces the opponent down to 3 cards. Value is
# entirely in the excess, so it scales with their pre-play hand size.
XEROSIC_PER_CARD_REWARD = 0.012
XEROSIC_MAX_REWARD = 0.080

# Eri discards up to 2 Items from the revealed hand.
ERI_PER_ITEM_REWARD = 0.015

# ── Targeting / tempo ─────────────────────────────────────────────────────
# Boss's Orders onto the Froslass line is the priority use: Freezing Shroud
# is the main way this deck loses, and 90/60 HP is inside Superb Scissors.
BOSS_FROSLASS_REWARD = 0.080
BOSS_KO_RANGE_REWARD = 0.030   # dragged up something Crustle's 120 can finish
BOSS_MULTI_PRIZE_REWARD = 0.030  # ...and it's worth 2+ prizes

# Switch used to unstick the Active while a powered (3+ energy) Crustle waits
# on the bench. Crustle's retreat cost is 3, so this is the swap.
SWITCH_UNSTICK_REWARD = 0.040

# Pokegear 3.0 hitting a Supporter we actually want right now (see
# _wanted_supporter_ids); a smaller amount for any Supporter at all.
POKEGEAR_WANTED_REWARD = 0.020
POKEGEAR_ANY_REWARD = 0.005

# Lillie's Determination shuffles our hand away for 6 (8 at exactly 6 prizes),
# so it is best on an empty hand and actively bad on a full one.
LILLIE_BASE_REWARD = 0.060
LILLIE_PER_CARD_DISCARDED = 0.010
LILLIE_MAX_PENALTY = -0.030

# Tutors used because the board actually needs them: Ultra Ball / Hilda when a
# Dwebble is stranded with no Crustle in hand, Petrel when the line isn't on
# the board at all or we need to dig for a heal.
TUTOR_ON_NEED_REWARD = 0.020

# ── Board shape ───────────────────────────────────────────────────────────
# Charged once, on the step the count crosses the cap (see docstring note 2).
MAX_KANGASKHAN_IN_PLAY = 1
MAX_CRUSTLE_LINE_IN_PLAY = 2
EXTRA_KANGASKHAN_PENALTY = 0.060
EXTRA_CRUSTLE_LINE_PENALTY = 0.030

# A fresh Kangaskhan hitting the board while the opponent has any Fighting
# Pokemon in play: 300 HP halves to an effective 150 against Fighting and it
# hands over 3 prizes. Largest single shaping penalty in the file.
KANGASKHAN_INTO_FIGHTING_PENALTY = 0.150

# Opening Active. Kangaskhan is the intended lead (300 HP body that draws 2 a
# turn while the bench develops); leading a Dwebble means a 60-70 HP starter
# in the firing line. Fires at most once per game.
LEAD_KANGASKHAN_REWARD = 0.050
LEAD_WRONG_PENALTY = 0.050

# Attacking with Kangaskhan (Rapid-Fire Combo, 200+ but a coin-flip chain)
# while a fully-fuelled Crustle was sitting there is usually the worse line --
# it exposes the 3-prize body instead of the wall.
KANGASKHAN_ATTACK_OVER_CRUSTLE_PENALTY = 0.020

# ── Run Errand ────────────────────────────────────────────────────────────
# Kangaskhan's whole job in the Active Spot is drawing 2 a turn for free. The
# generic `draw` term pays 0.002 for that, which badly undervalues the single
# most repeatable edge in the deck, so a missed activation is charged
# directly. See _turn_end_penalties for why the detection is deliberately
# biased toward false negatives.
RUN_ERRAND_MISS_PENALTY = 0.010
RUN_ERRAND_MIN_DRAWS = 2

# ── Setup tempo ───────────────────────────────────────────────────────────
# One-shot payout the first time a Crustle in play reaches attack cost, decayed
# by how long it took. Everything else in the file rewards the individual
# pieces (evolve, attach, cape); this rewards the assembled wall existing, and
# existing *early*, which is what actually decides games against aggro.
SETUP_TEMPO_BASE = 0.080
SETUP_TEMPO_DECAY_PER_TURN = 0.015

# ── Wall matchup ──────────────────────────────────────────────────────────
# Crustle attacking while the defender is a Pokemon ex is the deck's win
# condition in miniature: Mysterious Rock Inn means that defender cannot
# damage it back at all, so the trade is free. Needs CARD_DB to identify ex.
CRUSTLE_WALL_ATTACK_REWARD = 0.012

# ── Retreating ────────────────────────────────────────────────────────────
# Retreat cost is 3 for both Kangaskhan and Crustle, and paying it discards
# that much Energy -- manually retreating a built Crustle throws the whole
# investment away. Switch (and Petrel fetching Switch) is the correct swap,
# which is what SWITCH_UNSTICK_REWARD pays for; this is the other side of it.
RETREAT_ENERGY_PENALTY = 0.015

# ── Deck-out ──────────────────────────────────────────────────────────────
# A wall deck that plans to grind is the archetype most likely to lose to its
# own draw engine. Charged once on each threshold crossing, not per step.
DECK_LOW_THRESHOLD = 8
DECK_CRITICAL_THRESHOLD = 3
DECK_LOW_PENALTY = 0.040
DECK_CRITICAL_PENALTY = 0.120

# Boss's Orders is the turn's Supporter. Spending it to drag up something we
# can neither KO nor need gone is worse than not playing it.
BOSS_WASTED_PENALTY = 0.020

# ── Stadium ───────────────────────────────────────────────────────────────# Team Rocket's Factory (extra draw off our 4 Petrel), Community Center (heal
# 10 across the board after a Supporter -- with a 250+ HP Crustle that adds
# up), Festival Grounds (Special Condition immunity for anything with Energy,
# covering the status hole in Mysterious Rock Inn).
STADIUM_PLAY_REWARD = 0.030
STADIUM_BUMP_REWARD = 0.050

# ── Card IDs (from Card_ID_List_EN.pdf) ───────────────────────────────────
BASIC_GRASS_ENERGY = 1
MIST_ENERGY = 11               # TEF 161
SPIKY_ENERGY = 14              # JTG 159
GROW_GRASS_ENERGY = 18         # POR 86

SNORUNT_TWM = 103              # TWM 51
FROSLASS_TWM = 104             # TWM 53  -- Freezing Shroud
SNORUNT_ASC = 860              # ASC 46
DWEBBLE = 344                  # DRI 11
CRUSTLE = 345                  # DRI 12
MEGA_KANGASKHAN_EX = 756       # MEG 104

BUDDY_BUDDY_POFFIN = 1086      # TEF 144
CRUSHING_HAMMER = 1120         # SVI 168
ULTRA_BALL = 1121              # SVI 196
POKEGEAR_3 = 1122              # SVI 186
SWITCH = 1123                  # SVI 194
JUMBO_ICE_CREAM = 1147         # PFL 91
HEROS_CAPE = 1159              # TEF 152
BOSSS_ORDERS = 1182            # PAL 172
ERI = 1186                     # TEF 146
BIANCAS_DEVOTION = 1190        # TEF 142
XEROSICS_MACHINATIONS = 1197   # SFA 64
TEAM_ROCKETS_PETREL = 1219     # DRI 176
HILDA = 1225                   # WHT 84
LILLIES_DETERMINATION = 1227   # MEG 119
COMMUNITY_CENTER = 1242        # TWM 146
FESTIVAL_GROUNDS = 1245        # TWM 149
TEAM_ROCKETS_FACTORY = 1257    # DRI 173

CRUSTLE_LINE_IDS = (DWEBBLE, CRUSTLE)
FROSLASS_LINE_IDS = (FROSLASS_TWM, SNORUNT_TWM, SNORUNT_ASC)
STADIUM_IDS = (TEAM_ROCKETS_FACTORY, COMMUNITY_CENTER, FESTIVAL_GROUNDS)

# Superb Scissors damage, used for "can Crustle finish this?" checks. Ignores
# Weakness/Resistance, same approximation the starmie file made.
CRUSTLE_ATTACK_DAMAGE = 120
CRUSTLE_ATTACK_COST = 3        # {G}{C}{C} -- also crustle_agent.py's threshold

ENERGY_BONUS_BY_ID = {
    MIST_ENERGY: 0.012,        # highest: covers the effect-damage hole
    GROW_GRASS_ENERGY: 0.008,  # +20 HP each on the Grass wall
    SPIKY_ENERGY: 0.005,       # 2 counters back on the attacker
    BASIC_GRASS_ENERGY: 0.001,
}

try:
    from ptcg.api import all_card_data, CardType
    CARD_DB = {c.cardId: c for c in all_card_data()}
except Exception:
    CardType = None
    CARD_DB = {}

# Log entry "type" is an IntEnum (ptcg.api.LogType), NOT the string names
# vis.json renders -- see the starmie file's long comment on the multi-million
# step runs that silently logged 0.0 because of this. Numeric fallbacks are
# the documented ptcg/api.py values.
try:
    from ptcg.api import LogType as _LogType
    LOG_DRAW = int(_LogType.DRAW)
    LOG_ATTACH = int(_LogType.ATTACH)
    LOG_EVOLVE = int(_LogType.EVOLVE)
    LOG_ATTACK = int(_LogType.ATTACK)
    LOG_HP_CHANGE = int(_LogType.HP_CHANGE)
    LOG_TURN_END = int(_LogType.TURN_END)
except Exception:
    LOG_DRAW, LOG_ATTACH, LOG_EVOLVE, LOG_ATTACK, LOG_HP_CHANGE = 4, 11, 12, 15, 16
    LOG_TURN_END = 3

# EnergyType.FIGHTING. CardData.energyType comes back as a raw int, so this is
# compared numerically -- see _is_fighting_pokemon.
try:
    from ptcg.api import EnergyType as _EnergyType
    FIGHTING_ENERGY_TYPE = int(_EnergyType.FIGHTING)
except Exception:
    FIGHTING_ENERGY_TYPE = 6


# ── Shared accessors ──────────────────────────────────────────────────────

def _players(obs_dict):
    return (obs_dict.get("current") or {}).get("players") or [{}, {}]


def _player(obs_dict, player_index):
    players = _players(obs_dict)
    return (players[player_index] or {}) if player_index < len(players) else {}


def _energies(mon):
    """Energy *provided* by the cards attached to a Pokemon dict.

    CONFIRMED against a live obs: a Pokemon carries BOTH keys. `energies` is
    list[EnergyType] -- the energy the attachments actually provide, which is
    what an attack cost is paid with -- and `energyCards` is list[Card], the
    physical cards. Every caller here is asking "can this thing pay {G}{C}{C}
    yet", so `energies` is the one to prefer. Returns [] rather than raising.
    """
    if not isinstance(mon, dict):
        return []
    for key in ("energies", "energyCards", "energy"):
        val = mon.get(key)
        if val:
            return list(val)
    return []


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


def _has_any_in_play(obs_dict, card_ids, player_index):
    if not card_ids:
        return False
    return _count_any_in_play(obs_dict, player_index, card_ids) > 0


def _hand_ids(obs_dict, player_index):
    return [c.get("id") for c in (_player(obs_dict, player_index).get("hand") or []) if c]


def _hand_size(obs_dict, player_index):
    """Cards in hand. MUST read handCount, not len(hand).

    PlayerState.hand is `list[Card] | None` and is None for the OPPONENT -- we
    only ever see our own hand contents. len(hand or []) therefore returns 0
    for every opponent query, which silently zeroed reward/xerosic (its whole
    value is max(0, opp_hand - 3)) and killed the Xerosic/Eri branch of
    _wanted_supporter_ids. handCount is present for both players.
    """
    p = _player(obs_dict, player_index)
    count = p.get("handCount")
    if isinstance(count, int):
        return count
    return len(p.get("hand") or [])


def _prizes_remaining(obs_dict, player_index):
    return len(_player(obs_dict, player_index).get("prize") or [])


def _deck_remaining(obs_dict, player_index):
    """Cards left in deck. The field is deckCount -- PlayerState has no "deck"
    key at all (the only `deck` in the schema is SelectData.deck, the cards
    offered when searching). The old len(get("deck")) read 0 every step, so
    both deck-out thresholds were compared against a constant 0 and
    reward/deck_out could never fire."""
    p = _player(obs_dict, player_index)
    count = p.get("deckCount")
    if isinstance(count, int):
        return count
    return len(p.get("deck") or [])


def _newly_discarded_ids(prev_obs, cur_obs, player_index):
    """Card ids that entered player_index's discard since prev_obs (duplicates
    included). Count-based diff, so it makes no assumption about discard-pile
    ordering, and it is a far safer "was this card played" detector than
    guessing a Play log schema. It also fires on non-play discards (Ultra
    Ball's cost, energy knocked off by Crushing Hammer) -- an accepted
    approximation, which is why the Hammer term below cross-checks *whose*
    discard grew.
    """
    prev_ids = Counter(c.get("id") for c in (_player(prev_obs, player_index).get("discard") or []) if c)
    cur_ids = Counter(c.get("id") for c in (_player(cur_obs, player_index).get("discard") or []) if c)
    out = []
    for cid, cnt in cur_ids.items():
        out.extend([cid] * max(0, cnt - prev_ids.get(cid, 0)))
    return out


def _newly_in_hand_ids(prev_obs, cur_obs, player_index):
    """Card ids that appeared in our hand since prev_obs. Used to tell what a
    search card actually fetched (Pokegear/Petrel/Hilda)."""
    prev_ids = Counter(_hand_ids(prev_obs, player_index))
    cur_ids = Counter(_hand_ids(cur_obs, player_index))
    out = []
    for cid, cnt in cur_ids.items():
        out.extend([cid] * max(0, cnt - prev_ids.get(cid, 0)))
    return out


# ── Card-type predicates ──────────────────────────────────────────────────
# CARD_DB first, then a hardcoded fallback covering our own 60. The fallback
# exists because every one of these predicates gates a reward term, and with
# ptcg.api unimportable the CARD_DB-only version returns None for everything
# -- which silently zeroes the energy, supporter, Hammer and Ice Cream terms
# for the whole run. That is precisely the failure the starmie file's
# post-mortem describes, so the terms that only ever look at *our* cards are
# made independent of it. Terms that must classify an *opponent's* card
# (Eri's Item count, Hammer's "did they lose an Energy") still need CARD_DB
# and fail safe to 0 without it.

DECK_ENERGY_IDS = frozenset({BASIC_GRASS_ENERGY, MIST_ENERGY, SPIKY_ENERGY, GROW_GRASS_ENERGY})
DECK_SUPPORTER_IDS = frozenset({
    LILLIES_DETERMINATION, BOSSS_ORDERS, TEAM_ROCKETS_PETREL, HILDA, ERI,
    XEROSICS_MACHINATIONS, BIANCAS_DEVOTION,
})
DECK_ITEM_IDS = frozenset({
    JUMBO_ICE_CREAM, POKEGEAR_3, BUDDY_BUDDY_POFFIN, ULTRA_BALL, SWITCH,
    CRUSHING_HAMMER,
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
    # CardType has no ENERGY member -- it splits into BASIC_ENERGY (5) and
    # SPECIAL_ENERGY (6). The old getattr(CardType, "ENERGY", object()) always
    # resolved to a throwaway sentinel, so every BASIC energy card in the game
    # classified as "not energy": Crushing Hammer's did-they-actually-lose-an-
    # Energy check and the retreat-cost count both under-fired against decks
    # running basic Energy (i.e. nearly every opponent in the pool).
    return _classify(
        card_id,
        (
            getattr(CardType, "BASIC_ENERGY", object()),
            getattr(CardType, "SPECIAL_ENERGY", object()),
        ),
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


def _is_fighting_pokemon(card_id):
    """True if the card is a {F} Pokemon.

    CONFIRMED: the field is CardData.energyType, and to_dataclass leaves it as
    a RAW INT (EnergyType.FIGHTING == 6), not an enum member. The previous
    version stringified the value and looked for "FIGHT" in it, which for the
    int 6 gives "6" -- so the largest single penalty in this file
    (KANGASKHAN_INTO_FIGHTING_PENALTY) never once fired. Compare numerically;
    the name/str branches are kept only in case the DLL later hands back real
    enum members.
    """
    if not CARD_DB or card_id is None:
        return False
    data = CARD_DB.get(card_id)
    if data is None:
        return False
    # Only Pokemon have a meaningful energyType here -- a Fighting *Energy*
    # card carries energyType 6 too, and this must not call it a Pokemon.
    card_type = getattr(data, "cardType", None)
    if card_type is not None and CardType is not None and card_type != CardType.POKEMON:
        return False
    val = getattr(data, "energyType", None)
    if val is None:
        return False
    candidates = val if isinstance(val, (list, tuple, set)) else [val]
    for c in candidates:
        if isinstance(c, int) and c == FIGHTING_ENERGY_TYPE:
            return True
        name = str(getattr(c, "name", None) or c).upper()
        if name in ("F", "FIGHTING"):
            return True
    return False


# ── Log readers (schemas confirmed in the starmie file except where noted) ─

def _logs(obs_dict):
    return obs_dict.get("logs") or []


def _entries(obs_dict, log_type, player_index):
    return [e for e in _logs(obs_dict)
            if isinstance(e, dict)
            and e.get("type") == log_type
            and e.get("playerIndex") == player_index]


def _draw_count(obs_dict, player_index):
    total = 0
    for e in _entries(obs_dict, LOG_DRAW, player_index):
        amount = e.get("count", e.get("amount"))
        total += amount if isinstance(amount, (int, float)) else 1
    return total


def _evolve_count(obs_dict, player_index):
    return len(_entries(obs_dict, LOG_EVOLVE, player_index))


def _attach_events(obs_dict, player_index):
    """(attached_card_id, receiving_pokemon_card_id) for each Attach entry.

    Attach entries carry {cardId, serial, cardIdTarget, serialTarget}; tool
    attaches share the shape with energy attaches, so callers filter by the
    attached card's own type. cardIdTarget is the field the "which Pokemon
    got it" logic here depends on -- if it turns out absent, every target
    term degrades to 0 (unknown target) instead of misfiring.
    """
    return [(e.get("cardId"), e.get("cardIdTarget"))
            for e in _entries(obs_dict, LOG_ATTACH, player_index)]


def _hp_delta(obs_dict, player_index, positive):
    total = 0
    for e in _entries(obs_dict, LOG_HP_CHANGE, player_index):
        v = e.get("value")
        if isinstance(v, (int, float)) and ((v > 0) if positive else (v < 0)):
            total += v if positive else -v
    return total


def _damage_dealt(obs_dict, target_player_index):
    """HP lost across the target's whole board (bench damage included)."""
    return _hp_delta(obs_dict, target_player_index, positive=False)


def _heal_dealt(obs_dict, player_index):
    return _hp_delta(obs_dict, player_index, positive=True)


def _attack_id_used(obs_dict, player_index):
    entries = _entries(obs_dict, LOG_ATTACK, player_index)
    return entries[0].get("attackId") if entries else None


def _attacker_id(obs_dict, player_index):
    """Card id of the Pokemon that attacked, if the Attack entry carries one.

    `cardId` on Attack entries is the same field the starmie agent reads for
    its Itchy Pollen check, so the field exists; that it holds the *attacker*
    rather than the target is the inference. Falls back to the Active.
    """
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


# ── Board-state helpers specific to this deck ─────────────────────────────

def _crustle_line_count(obs_dict, player_index):
    return _count_any_in_play(obs_dict, player_index, CRUSTLE_LINE_IDS)


def _best_crustle_energy(obs_dict, player_index):
    """Most energy on any Crustle we have in play (-1 if we have none)."""
    best = -1
    for m in _pokemon_in_play(obs_dict, player_index):
        if m.get("id") == CRUSTLE:
            best = max(best, len(_energies(m)))
    return best


def _ready_crustle_on_bench(obs_dict, player_index):
    return any(m.get("id") == CRUSTLE and len(_energies(m)) >= CRUSTLE_ATTACK_COST
               for m in _bench_pokemon(obs_dict, player_index))


def _line_member_wants_energy(obs_dict, player_index):
    """Some Dwebble/Crustle in play that isn't yet at attack cost -- i.e. an
    attachment target that beats putting the energy on Kangaskhan."""
    for m in _pokemon_in_play(obs_dict, player_index):
        if m.get("id") in CRUSTLE_LINE_IDS and len(_energies(m)) < CRUSTLE_ATTACK_COST:
            return True
    return False


def _stranded_dwebble(obs_dict, player_index):
    """A Dwebble in play with no Crustle in hand to evolve it -- the trigger
    for wanting Ultra Ball / Hilda."""
    return (_count_in_play(obs_dict, player_index, DWEBBLE) > 0
            and CRUSTLE not in _hand_ids(obs_dict, player_index))


def _wants_ice_cream(obs_dict, player_index):
    """Active is damaged enough for a near-full Jumbo Ice Cream and has the
    3 energy the card requires, but we don't hold one."""
    mon = _active_pokemon(obs_dict, player_index)
    if not mon or len(_energies(mon)) < 3:
        return False
    hp, max_hp = mon.get("hp"), mon.get("maxHp")
    if not isinstance(hp, (int, float)) or not isinstance(max_hp, (int, float)):
        return False
    return (max_hp - hp) >= 60 and JUMBO_ICE_CREAM not in _hand_ids(obs_dict, player_index)


def _wanted_supporter_ids(obs_dict, me_index, opp_index):
    """Supporters worth digging for *right now*, used to grade Pokegear hits.

    Boss when there's a bench to drag from (and it's near-mandatory when the
    Froslass line is back there), Hilda when a Dwebble is stranded or a
    Crustle still needs energy, Lillie when we're out of cards, Xerosic/Eri
    when the opponent is sitting on a hand, Petrel when the board is missing
    the line or we need to dig for a heal.
    """
    wanted = set()
    if _bench_pokemon(obs_dict, opp_index):
        wanted.add(BOSSS_ORDERS)
    if _stranded_dwebble(obs_dict, me_index) or _line_member_wants_energy(obs_dict, me_index):
        wanted.add(HILDA)
    if _hand_size(obs_dict, me_index) <= 3:
        wanted.add(LILLIES_DETERMINATION)
    if _hand_size(obs_dict, opp_index) >= 6:
        wanted.add(XEROSICS_MACHINATIONS)
        wanted.add(ERI)
    if _crustle_line_count(obs_dict, me_index) == 0 or _wants_ice_cream(obs_dict, me_index):
        wanted.add(TEAM_ROCKETS_PETREL)
    return wanted


# ── Per-game state (must be reset between games) ──────────────────────────

_turns_taken = {0: 0, 1: 0}
_lead_scored = {0: False, 1: False}
_setup_scored = {0: False, 1: False}
_turn_draws = {0: 0, 1: 0}
_turn_kangaskhan_active = {0: False, 1: False}
_turn_attacked = {0: False, 1: False}


def reset_turn_tracking():
    """Call at the start of every game, both self-play and heuristic-opponent
    mode. Only correct for one game per process at a time -- key this by env
    id if games ever run concurrently through a shared process.

    CabtEnv.reset() calls this. It is not optional: _lead_scored and
    _setup_scored are one-shot latches, so a worker that never resets them
    pays `lead` and `setup_tempo` on its first episode and 0.0 for every
    episode after that.
    """
    global _turns_taken, _lead_scored, _setup_scored, _turn_draws
    global _turn_kangaskhan_active, _turn_attacked
    _turns_taken = {0: 0, 1: 0}
    _lead_scored = {0: False, 1: False}
    _setup_scored = {0: False, 1: False}
    _turn_draws = {0: 0, 1: 0}
    _turn_kangaskhan_active = {0: False, 1: False}
    _turn_attacked = {0: False, 1: False}


# Alias: this now resets lead, setup and per-turn accumulators too, not just
# turn counts. Anything that used to call reset_turn_tracking() still works.
reset_game_state = reset_turn_tracking


def _turn_end_penalties(prev_obs, cur_obs, me_index):
    """(no_attack, run_errand_missed) -- both only fire on turn hand-off.

    Accumulates three things across the steps of our turn and settles them on
    the step whose log window contains our TURN_END: whether we attacked at
    all, how many cards we drew, and whether Kangaskhan held the Active Spot.

    Hand-off is detected from the TURN_END log entry, NOT from a
    current.yourIndex flip. The yourIndex test never fired in the real
    training env: CabtEnv auto-plays the opponent through
    _play_opponent_until_learner_turn, so every observation the learner is
    scored on already has yourIndex back on the learner and the whole branch
    was dead -- reward/no_attack and reward/run_errand logged a flat 0.0 for
    entire runs. `logs` is a per-viewer window (the engine reports each event
    exactly once to each player), so our TURN_END lands in exactly one of our
    steps even when that step spans the opponent's entire turn.

    `attacked` likewise has to accumulate: cur_obs's log window only covers
    the events since our previous decision, so checking it alone charges
    NO_ATTACK_PENALTY on every turn where the attack wasn't the very last
    thing we did.

    The Run Errand check is deliberately conservative. Draws from Lillie,
    Petrel-into-a-draw-card or Team Rocket's Factory all land in the same
    counter, so a turn that skipped the Ability but drew off a Supporter looks
    identical to one that used it -- the threshold of 2 means we under-charge
    (miss real misses) rather than punish turns that did use it. Raise
    RUN_ERRAND_MIN_DRAWS to 3 only after confirming the turn-start draw is
    logged with our playerIndex.

    Must be called exactly once per step: it mutates the accumulators.
    """
    _turn_draws[me_index] = _turn_draws.get(me_index, 0) + _draw_count(cur_obs, me_index)
    if _entries(cur_obs, LOG_ATTACK, me_index):
        _turn_attacked[me_index] = True
    active = _active_pokemon(prev_obs, me_index)
    if active and active.get("id") == MEGA_KANGASKHAN_EX:
        _turn_kangaskhan_active[me_index] = True

    if not _entries(cur_obs, LOG_TURN_END, me_index):
        return 0.0, 0.0

    _turns_taken[me_index] = _turns_taken.get(me_index, 0) + 1
    no_attack = (NO_ATTACK_PENALTY
                 if (_turns_taken[me_index] >= 2 and not _turn_attacked[me_index])
                 else 0.0)

    run_errand = 0.0
    if (_turn_kangaskhan_active[me_index]
            and _turn_draws[me_index] < RUN_ERRAND_MIN_DRAWS):
        run_errand = RUN_ERRAND_MISS_PENALTY

    _turn_draws[me_index] = 0
    _turn_kangaskhan_active[me_index] = False
    _turn_attacked[me_index] = False
    return no_attack, run_errand


def _lead_reward(prev_obs, cur_obs, me_index):
    """One-shot bonus for the opening Active being Kangaskhan.

    Grades the *earliest* board state this function sees: prev_obs's Active
    if there is one, otherwise the Active that appeared in cur_obs (the
    placement transition). Fires at most once per game, so it needs
    reset_turn_tracking() between games like the turn counter does. It does
    not try to detect SelectContext.TO_ACTIVE -- the reward function is only
    handed board states, not the select context.
    """
    if _lead_scored.get(me_index):
        return 0.0
    mon = _active_pokemon(prev_obs, me_index) or _active_pokemon(cur_obs, me_index)
    if not mon:
        return 0.0
    _lead_scored[me_index] = True
    return LEAD_KANGASKHAN_REWARD if mon.get("id") == MEGA_KANGASKHAN_EX else -LEAD_WRONG_PENALTY


# ── Composite term helpers ────────────────────────────────────────────────

def _energy_terms(cur_obs, me_index):
    """(attach_reward, target_reward) over this step's Attach events.

    attach_reward pays for using the attachment at all plus a per-energy-card
    bonus; target_reward grades where it landed. Energy on Kangaskhan is only
    charged when a line member was actually available to take it -- if
    Kangaskhan is all we have, fuelling it is correct.
    """
    attach_reward = 0.0
    target_reward = 0.0
    line_available = _line_member_wants_energy(cur_obs, me_index)

    for card_id, target_id in _attach_events(cur_obs, me_index):
        if not _is_energy_card(card_id):
            continue
        attach_reward += ENERGY_ATTACH_REWARD + ENERGY_BONUS_BY_ID.get(card_id, 0.0)
        if target_id in CRUSTLE_LINE_IDS:
            target_reward += ENERGY_ON_CRUSTLE_LINE_REWARD
        elif target_id == MEGA_KANGASKHAN_EX and line_available:
            target_reward -= ENERGY_ON_KANGASKHAN_PENALTY
    return attach_reward, target_reward


def _hero_cape_reward(cur_obs, me_index):
    total = 0.0
    for card_id, target_id in _attach_events(cur_obs, me_index):
        if card_id != HEROS_CAPE:
            continue
        if target_id == CRUSTLE:
            total += HERO_CAPE_ON_CRUSTLE_REWARD
        elif target_id is not None:
            total -= HERO_CAPE_WASTED_PENALTY
    return total


def _heal_reward(prev_obs, cur_obs, me_index, my_discards):
    """Ice Cream gets the threshold treatment (see ICE_CREAM_BASE); other
    healing is small and linear.

    Attribution is by "was a Jumbo Ice Cream discarded this step", so a step
    that both Ice Creams and heals some other way lumps the total under the
    Ice Cream branch. Acceptable: those effects don't stack often here.
    """
    healed = _heal_dealt(cur_obs, me_index)
    if healed <= 0:
        return 0.0
    if JUMBO_ICE_CREAM in my_discards:
        return ICE_CREAM_BASE + ICE_CREAM_PER_10HP * (healed / 10)
    return HEAL_REWARD_PER_10HP * (healed / 10)


def _hammer_reward(prev_obs, cur_obs, me_index, opp_index, my_discards):
    if CRUSHING_HAMMER not in my_discards:
        return 0.0
    reward = HAMMER_PLAY_REWARD
    opp_lost_energy = any(_is_energy_card(cid)
                          for cid in _newly_discarded_ids(prev_obs, cur_obs, opp_index))
    if opp_lost_energy:
        reward += HAMMER_LAND_REWARD
    return reward


def _boss_reward(prev_obs, cur_obs, me_index, opp_index, my_discards):
    """Grade a Boss's Orders by what it dragged into the Active Spot.

    Reads the post-effect board only: no credit assignment to whichever later
    attack actually converts the KO, and the 120 threshold ignores Weakness
    and Resistance. The Froslass branch checks the line was on the *bench*
    beforehand so it doesn't pay out for a Froslass that was already Active.
    """
    if BOSSS_ORDERS not in my_discards:
        return 0.0
    target = _active_pokemon(cur_obs, opp_index)
    if not target:
        return 0.0

    reward = 0.0
    froslass_was_benched = any(m.get("id") in FROSLASS_LINE_IDS
                               for m in _bench_pokemon(prev_obs, opp_index))
    if target.get("id") in FROSLASS_LINE_IDS and froslass_was_benched:
        reward += BOSS_FROSLASS_REWARD

    hp = target.get("hp")
    if isinstance(hp, (int, float)) and hp <= CRUSTLE_ATTACK_DAMAGE:
        reward += BOSS_KO_RANGE_REWARD
        if _prize_value_by_id(target.get("id")) >= 2:
            reward += BOSS_MULTI_PRIZE_REWARD

    # Boss is the turn's Supporter. Dragging up something we can neither
    # finish nor need gone costs us Lillie/Hilda/Petrel for the turn.
    if reward == 0.0:
        reward -= BOSS_WASTED_PENALTY
    return reward


def _switch_reward(prev_obs, cur_obs, me_index, my_discards):
    """Switch played while the Active was stuck and a fuelled Crustle waited
    on the bench. 'Stuck' = the Active wasn't already a Crustle with enough
    energy to attack, which covers both the Kangaskhan-swap and the
    dragged-up-Dwebble case."""
    if SWITCH not in my_discards:
        return 0.0
    if not _ready_crustle_on_bench(prev_obs, me_index):
        return 0.0
    mon = _active_pokemon(prev_obs, me_index)
    active_is_ready_crustle = (
        mon is not None
        and mon.get("id") == CRUSTLE
        and len(_energies(mon)) >= CRUSTLE_ATTACK_COST
    )
    return 0.0 if active_is_ready_crustle else SWITCH_UNSTICK_REWARD


def _pokegear_reward(prev_obs, cur_obs, me_index, opp_index, my_discards):
    if POKEGEAR_3 not in my_discards:
        return 0.0
    wanted = _wanted_supporter_ids(prev_obs, me_index, opp_index)
    gained = _newly_in_hand_ids(prev_obs, cur_obs, me_index)
    supporters = [cid for cid in gained if _is_supporter_card(cid)]
    if not supporters:
        return 0.0
    return POKEGEAR_WANTED_REWARD if any(cid in wanted for cid in supporters) else POKEGEAR_ANY_REWARD


def _lillie_reward(prev_obs, cur_obs, me_index, my_discards):
    """Lillie shuffles the hand away for 6 (8 at exactly 6 Prizes), so the
    cost is every card it throws away. prev hand size includes Lillie itself,
    hence the -1."""
    if LILLIES_DETERMINATION not in my_discards:
        return 0.0
    discarded = max(0, _hand_size(prev_obs, me_index) - 1)
    reward = LILLIE_BASE_REWARD - LILLIE_PER_CARD_DISCARDED * discarded
    return max(LILLIE_MAX_PENALTY, reward)


def _tutor_reward(prev_obs, cur_obs, me_index, my_discards):
    """Ultra Ball / Hilda into a stranded Dwebble, Petrel when the board has
    no line at all or we need to dig for a heal."""
    reward = 0.0
    stranded = _stranded_dwebble(prev_obs, me_index)
    for cid in my_discards:
        if cid in (ULTRA_BALL, HILDA) and stranded:
            reward += TUTOR_ON_NEED_REWARD
        elif cid == TEAM_ROCKETS_PETREL:
            if _crustle_line_count(prev_obs, me_index) == 0 or _wants_ice_cream(prev_obs, me_index):
                reward += TUTOR_ON_NEED_REWARD
        elif cid == BUDDY_BUDDY_POFFIN and _crustle_line_count(prev_obs, me_index) == 0:
            reward += TUTOR_ON_NEED_REWARD
    return reward


def _board_shape_penalty(prev_obs, cur_obs, me_index):
    """Charged on the transition that pushes the board past the caps, NOT per
    step while it's over (see module docstring, note 2)."""
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


def _kangaskhan_into_fighting_penalty(prev_obs, cur_obs, me_index, opp_index):
    """A new Kangaskhan hitting the board while the opponent shows Fighting.
    Silently 0 if _is_fighting_pokemon can't resolve a type field."""
    if _count_in_play(cur_obs, me_index, MEGA_KANGASKHAN_EX) <= _count_in_play(
            prev_obs, me_index, MEGA_KANGASKHAN_EX):
        return 0.0
    fighting = any(_is_fighting_pokemon(m.get("id"))
                   for m in _pokemon_in_play(prev_obs, opp_index))
    return KANGASKHAN_INTO_FIGHTING_PENALTY if fighting else 0.0


def _attacker_choice_penalty(prev_obs, cur_obs, me_index):
    """Attacking with Kangaskhan while a ready Crustle sat on the bench."""
    if _attacker_id(cur_obs, me_index) != MEGA_KANGASKHAN_EX:
        return 0.0
    return (KANGASKHAN_ATTACK_OVER_CRUSTLE_PENALTY
            if _ready_crustle_on_bench(prev_obs, me_index) else 0.0)


def _setup_tempo_reward(cur_obs, me_index):
    """One-shot payout the first time we get a Crustle to attack cost, decayed
    by the number of our turns that have already ended. Fires at most once per
    game; a Crustle that is knocked out and rebuilt doesn't pay again, which
    is intended -- this rewards reaching the plan, not maintaining it."""
    if _setup_scored.get(me_index):
        return 0.0
    if _best_crustle_energy(cur_obs, me_index) < CRUSTLE_ATTACK_COST:
        return 0.0
    _setup_scored[me_index] = True
    decay = SETUP_TEMPO_DECAY_PER_TURN * _turns_taken.get(me_index, 0)
    return max(0.0, SETUP_TEMPO_BASE - decay)


def _is_ex_pokemon(card_id):
    """True for ex / Mega Evolution ex. CARD_DB-only: _prize_value_by_id
    defaults everything to 1 prize when the DB is unavailable, so this term
    goes quiet rather than misfiring."""
    return _prize_value_by_id(card_id) >= 2


def _wall_matchup_reward(prev_obs, cur_obs, me_index, opp_index):
    """Crustle attacking into a Pokemon ex: Mysterious Rock Inn means that
    defender cannot damage it back, so it's a free hit. Reads the defender
    from prev_obs, since the KO may have already cleared the Active Spot by
    the time cur_obs is taken."""
    if _attacker_id(cur_obs, me_index) != CRUSTLE:
        return 0.0
    defender = _active_pokemon(prev_obs, opp_index)
    if not defender:
        return 0.0
    return CRUSTLE_WALL_ATTACK_REWARD if _is_ex_pokemon(defender.get("id")) else 0.0


def _active_identity(obs_dict, player_index):
    """Serial if the observation carries one, else the card id. Serial matters
    here because two copies of the same Pokemon in the Active Spot across a
    step would otherwise look like no change at all."""
    mon = _active_pokemon(obs_dict, player_index)
    if not mon:
        return None
    return mon.get("serial", mon.get("id"))


def _retreat_penalty(prev_obs, cur_obs, me_index, my_discards, opp_took):
    """Charge for Energy dumped paying a retreat cost.

    INFERRED, not confirmed: there is no Retreat log type in the set this file
    reads, so a retreat is inferred from "our Active changed, our own Energy
    hit the discard, no Switch was played, and the opponent took no Prize".
    The Prize guard is what separates a retreat from a knockout (a KO also
    sends the Active's Energy to our discard). A Boss's Orders or Switch
    played *by the opponent* to drag something up would also move our Active
    without a retreat, but that doesn't discard our Energy, so it won't fire.
    """
    if opp_took > 0 or SWITCH in my_discards:
        return 0.0
    if _active_identity(prev_obs, me_index) == _active_identity(cur_obs, me_index):
        return 0.0
    energies_discarded = sum(1 for cid in my_discards if _is_energy_card(cid))
    return RETREAT_ENERGY_PENALTY * energies_discarded


def _deck_out_penalty(prev_obs, cur_obs, me_index):
    """Charged once on each threshold crossing, not per step while thin.

    A grind deck that plans to go long is the archetype most likely to draw
    itself out, and nothing else in this file pushes back on over-drawing --
    the `draw` term pays for it. Crossing is one-way here: a Lillie that
    shuffles a big hand back can lift the count above the threshold again and
    re-arm the charge, which is correct (it really did buy deck back).
    """
    prev_deck = _deck_remaining(prev_obs, me_index)
    cur_deck = _deck_remaining(cur_obs, me_index)
    penalty = 0.0
    if prev_deck > DECK_LOW_THRESHOLD >= cur_deck:
        penalty += DECK_LOW_PENALTY
    if prev_deck > DECK_CRITICAL_THRESHOLD >= cur_deck:
        penalty += DECK_CRITICAL_PENALTY
    return penalty


def _stadium_reward(prev_obs, cur_obs, me_index, opp_index):
    prev_stadium = _stadium_identity(prev_obs)
    cur_stadium = _stadium_identity(cur_obs)
    if cur_stadium is None or cur_stadium[1] != me_index or cur_stadium == prev_stadium:
        return 0.0
    reward = STADIUM_PLAY_REWARD
    if prev_stadium is not None and prev_stadium[1] == opp_index:
        reward += STADIUM_BUMP_REWARD
    return reward


def _eri_reward(prev_obs, cur_obs, me_index, opp_index, my_discards):
    """Eri discards up to 2 Items off the revealed hand -- paid per Item that
    actually reached their discard this step."""
    if ERI not in my_discards:
        return 0.0
    items = sum(1 for cid in _newly_discarded_ids(prev_obs, cur_obs, opp_index)
                if _is_item_card(cid))
    return ERI_PER_ITEM_REWARD * min(items, 2)


def _xerosic_reward(prev_obs, me_index, opp_index, my_discards):
    """Xerosic cuts the opponent to 3 cards, so the value is the excess."""
    if XEROSICS_MACHINATIONS not in my_discards:
        return 0.0
    excess = max(0, _hand_size(prev_obs, opp_index) - 3)
    return min(XEROSIC_MAX_REWARD, XEROSIC_PER_CARD_REWARD * excess)


def _supporter_played_count(prev_obs, cur_obs, player_index):
    return sum(1 for cid in _newly_discarded_ids(prev_obs, cur_obs, player_index)
               if _is_supporter_card(cid))


# Fixed key set so the TensorBoard logger writes a COMPLETE series every
# rollout -- a term whose schema assumptions are wrong then shows up as a
# flat 0.0 line instead of silently never appearing at all.
REWARD_TERMS = (
    "terminal",
    "prize_mine",
    "prize_opp",
    "damage",
    "draw",
    "energy_attach",
    "energy_target",
    "evolve",
    "crustle_evolve",
    "hero_cape",
    "heal",
    "hammer",
    "xerosic",
    "eri",
    "boss",
    "switch_stuck",
    "pokegear",
    "lillie",
    "tutor",
    "stadium",
    "supporter",
    "board_shape",
    "kangaskhan_fighting",
    "attacker_choice",
    "wall_matchup",
    "setup_tempo",
    "retreat",
    "deck_out",
    "run_errand",
    "lead",
    "no_attack",
)


def reward_terms(prev_obs, cur_obs, done, result, me_index):
    """Named breakdown of compute_reward's terms, keyed for per-component
    logging (reward/<key> series).

    A terminal step returns a single {"terminal": +-1.0} entry -- there is
    nothing to shape once the game is over. Not side-effect free: it advances
    the per-player turn and lead counters, so call it once per step and sum
    the result rather than calling it and compute_reward both.
    """
    if done:
        return {"terminal": 1.0 if result == me_index else -1.0}

    opp_index = 1 - me_index
    my_discards = _newly_discarded_ids(prev_obs, cur_obs, me_index)

    my_took = max(0, _prizes_remaining(prev_obs, me_index) - _prizes_remaining(cur_obs, me_index))
    opp_took = max(0, _prizes_remaining(prev_obs, opp_index) - _prizes_remaining(cur_obs, opp_index))
    my_prize_reward = PRIZE_REWARD * my_took
    if my_took >= 2:
        my_prize_reward *= MULTI_PRIZE_MULTIPLIER

    energy_attach, energy_target = _energy_terms(cur_obs, me_index)

    crustle_evolved = max(0, _count_in_play(cur_obs, me_index, CRUSTLE)
                          - _count_in_play(prev_obs, me_index, CRUSTLE))

    # Settles the per-turn accumulators; must be called exactly once per step.
    no_attack, run_errand = _turn_end_penalties(prev_obs, cur_obs, me_index)

    terms = {
        "prize_mine": my_prize_reward,
        "prize_opp": -PRIZE_REWARD * opp_took,
        "damage": DAMAGE_REWARD_PER_100 * (_damage_dealt(cur_obs, opp_index) / 100),
        "draw": DRAW_REWARD_PER_CARD * _draw_count(cur_obs, me_index),
        "energy_attach": energy_attach,
        "energy_target": energy_target,
        "evolve": EVOLVE_REWARD * _evolve_count(cur_obs, me_index),
        # Counted off the board diff rather than the Evolve log's cardId /
        # cardIdTarget pairing, whose direction isn't confirmed.
        "crustle_evolve": CRUSTLE_EVOLVE_REWARD * crustle_evolved,
        "hero_cape": _hero_cape_reward(cur_obs, me_index),
        "heal": _heal_reward(prev_obs, cur_obs, me_index, my_discards),
        "hammer": _hammer_reward(prev_obs, cur_obs, me_index, opp_index, my_discards),
        "xerosic": _xerosic_reward(prev_obs, me_index, opp_index, my_discards),
        "eri": _eri_reward(prev_obs, cur_obs, me_index, opp_index, my_discards),
        "boss": _boss_reward(prev_obs, cur_obs, me_index, opp_index, my_discards),
        "switch_stuck": _switch_reward(prev_obs, cur_obs, me_index, my_discards),
        "pokegear": _pokegear_reward(prev_obs, cur_obs, me_index, opp_index, my_discards),
        "lillie": _lillie_reward(prev_obs, cur_obs, me_index, my_discards),
        "tutor": _tutor_reward(prev_obs, cur_obs, me_index, my_discards),
        "stadium": _stadium_reward(prev_obs, cur_obs, me_index, opp_index),
        "supporter": SUPPORTER_PLAY_REWARD * _supporter_played_count(prev_obs, cur_obs, me_index),
        "board_shape": -_board_shape_penalty(prev_obs, cur_obs, me_index),
        "kangaskhan_fighting": -_kangaskhan_into_fighting_penalty(
            prev_obs, cur_obs, me_index, opp_index),
        "attacker_choice": -_attacker_choice_penalty(prev_obs, cur_obs, me_index),
        "wall_matchup": _wall_matchup_reward(prev_obs, cur_obs, me_index, opp_index),
        "setup_tempo": _setup_tempo_reward(cur_obs, me_index),
        "retreat": -_retreat_penalty(prev_obs, cur_obs, me_index, my_discards, opp_took),
        "deck_out": -_deck_out_penalty(prev_obs, cur_obs, me_index),
        "run_errand": -run_errand,
        "lead": _lead_reward(prev_obs, cur_obs, me_index),
        "no_attack": -no_attack,
    }

    if SHAPING_SCALE != 1.0:
        terms = {k: v * SHAPING_SCALE for k, v in terms.items()}
    return terms


def compute_reward(prev_obs, cur_obs, done, result, me_index):
    """Total reward from the acting player's perspective -- the sum of
    reward_terms() (see it for the term list and the arguments).

    Single-value entry point for callers that don't need the breakdown.
    Callers that want both (CabtEnv, shipping terms out for TensorBoard)
    should call reward_terms() once and sum it, since reward_terms() is not
    side-effect free.
    """
    return sum(reward_terms(prev_obs, cur_obs, done, result, me_index).values())
