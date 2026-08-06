"""Mega Kangaskhan ex / Crustle heuristic.

Pilots the 60 cards in this package's deck.csv (a copy of the repo-root
crustle_deck.csv). Replaced the original hand-written version, which it beat
79% over 100 mirror games (79W-21L, seating alternated) -- same decklist on
both sides, so that number is piloting alone. `git log` this file for the
version it replaced.

Structure is borrowed from dashimaki360's public "Beating the Day-1 #1 Crustle
Bot" notebook (kaggle.com/code/dashimaki360/beating-the-day-1-1-crustle-bot):
score every option in `select.option`, sort descending, return the top indices
while honouring minCount/maxCount. That notebook's one big idea is the whole
skeleton here --

    do ALL your setup first, attack last, and always return a *valid* choice
    for any forced sub-selection

-- because attacking ends the turn, so every point of score above the attack
band is an action you get for free first.

What this file adds on top of that skeleton is the deck plan encoded in
training/crustle_rewards.py, expressed as scores instead of rewards:

  * energy goes on the Crustle line, never on Kangaskhan while a line member
    still wants it (Kangaskhan is a 3-prize liability);
  * Dwebble -> Crustle is close to the highest-value action in the game;
  * Hero's Cape belongs on the Crustle line (it survives the evolution, so
    caping a Dwebble is fine);
  * Jumbo Ice Cream is only played near its full 80;
  * Boss's Orders prioritises the Froslass line, then KO range, then prizes;
  * Switch (or Petrel fetching Switch) is the swap -- retreating a built
    Crustle discards 3 Energy and throws the investment away;
  * exactly 1 Kangaskhan + at most 2 Crustle-line members on board;
  * Kangaskhan leads (300 HP drawing 2 a turn), Crustle finishes.

GRASS AWARENESS (the thing the v1 agent gets wrong)
---------------------------------------------------
Superb Scissors costs {G}{C}{C}. The deck runs 13 Energy but only *five* of
them provide {G}: 4 Grow Grass + 1 Basic {G}. Mist and Spiky are both {C}. So
`len(energies) >= 3` is NOT the same as "Crustle can attack" -- a Crustle
holding Mist/Mist/Spiky is a 3-energy brick. `_can_attack` checks for a {G}
source, and the attach scoring routes a Grass source to the build target
first, then switches to preferring Mist (which plugs the effect-damage hole in
Mysterious Rock Inn) once the {G} requirement is covered.

TWO NAMED THREATS
-----------------
  * Froslass (104) / Snorunt (103, 860): Freezing Shroud puts a damage counter
    on every Pokemon with an Ability each Checkup. Crustle has an Ability, so
    Froslass chips straight through the ex-damage wall -- the one card that
    beats the gameplan by ignoring it. At 90/60/70 HP it is inside Superb
    Scissors, so Boss's Orders onto it is the priority use of the card, and
    Battle Cage is the backstop: it blanks Freezing Shroud for everything on
    our Bench (though NOT the Active Spot, so it mitigates rather than solves).
  * Fighting Pokemon: Kangaskhan is {C} with Fighting x2 weakness and gives up
    3 prizes, so a fresh one into a Fighting board is heavily suppressed.

SCORE BANDS (keep new constants inside these)
---------------------------------------------
    3000        Run Errand (free draw 2, always first)
    2000-2600   high-priority setup: evolve to Crustle, cape it, Switch to a
                ready one, Boss with a real target
    1000-1900   ordinary setup: items, supporters, secondary attachments
     600- 999   playable but low value
     300- 500   attacks (they end the turn, so they sit under all setup)
         100    END
          < 0   suppressed -- only returned when minCount forces it
"""

import os as _os

from ptcg.api import (
    Observation,
    to_observation_class,
    OptionType,
    SelectContext,
    AreaType,
    EnergyType,
    Pokemon,
    Card,
    State,
    PlayerState,
    SelectData,
)

# ── Card IDs (Card_ID_List_EN.pdf) ────────────────────────────────────────
BASIC_GRASS_ENERGY = 1
MIST_ENERGY = 11               # TEF 161 -- {C}, blanks attack *effects*
SPIKY_ENERGY = 14              # JTG 159 -- {C}, 2 counters back
GROW_GRASS_ENERGY = 18         # POR 86  -- {G}, +20 HP on a {G} Pokemon

SNORUNT_TWM = 103
FROSLASS_TWM = 104             # Freezing Shroud
SNORUNT_ASC = 860
DWEBBLE = 344
CRUSTLE = 345
MEGA_KANGASKHAN_EX = 756

BUDDY_BUDDY_POFFIN = 1086
HAND_TRIMMER = 1087            # both players discard down to 5, THEM first
CRUSHING_HAMMER = 1120
ULTRA_BALL = 1121
POKEGEAR_3 = 1122
SWITCH = 1123
JUMBO_ICE_CREAM = 1147
HEROS_CAPE = 1159
BOSSS_ORDERS = 1182
XEROSICS_MACHINATIONS = 1197
TEAM_ROCKETS_PETREL = 1219
HILDA = 1225
LILLIES_DETERMINATION = 1227
COMMUNITY_CENTER = 1242
FESTIVAL_GROUNDS = 1245
TEAM_ROCKETS_FACTORY = 1257
BATTLE_CAGE = 1264             # no damage counters onto BENCHED Pokemon

CRUSTLE_LINE_IDS = (DWEBBLE, CRUSTLE)
# Only card 104 actually has Freezing Shroud. Both Snorunts can evolve into it
# -- but they can equally become Mega Froslass ex (861), which has no Ability
# that touches us, so they are a maybe-threat and scored as one.
FROSLASS_PRE_EVOLUTION_IDS = (SNORUNT_TWM, SNORUNT_ASC)
FROSLASS_LINE_IDS = (SNORUNT_TWM, FROSLASS_TWM, SNORUNT_ASC)
STADIUM_IDS = (
    TEAM_ROCKETS_FACTORY, COMMUNITY_CENTER, FESTIVAL_GROUNDS, BATTLE_CAGE,
)

# Supporters in the current 60. Used to grade what Pokegear 3.0 digs up.
DECK_SUPPORTER_IDS = (
    BOSSS_ORDERS, HILDA, TEAM_ROCKETS_PETREL, LILLIES_DETERMINATION,
    XEROSICS_MACHINATIONS,
)
ENERGY_IDS = (BASIC_GRASS_ENERGY, MIST_ENERGY, SPIKY_ENERGY, GROW_GRASS_ENERGY)
GRASS_SOURCE_IDS = (BASIC_GRASS_ENERGY, GROW_GRASS_ENERGY)

# Attack IDs, read off all_attack().
ASCENSION = 478                # Dwebble  {C}   -- fetch Crustle from deck
SUPERB_SCISSORS = 479          # Crustle  {G}{C}{C} 120
RAPID_FIRE_COMBO = 1092        # Kangaskhan {C}{C}{C} 200+

CRUSTLE_ATTACK_DAMAGE = 120
KANGASKHAN_ATTACK_DAMAGE = 200
CRUSTLE_ATTACK_COST = 3
MAX_USEFUL_ENERGY = 4          # don't stack a 5th Energy on anything
MAX_CRUSTLE_LINE_IN_PLAY = 2
MAX_KANGASKHAN_IN_PLAY = 1


# ── Deck loading ──────────────────────────────────────────────────────────
def _read_deck() -> list[int]:
    """deck.csv shipped next to this module, with the Kaggle submission path
    and the project-root crustle_deck.csv as fallbacks.

    Deliberately no bare cwd-relative "deck.csv" in this list: the version
    this file replaced loaded exactly that, which from the repo root is a
    completely different archetype's list. It only surfaces on the
    deck-submission call, so it fails silently rather than loudly. Better to
    raise than to hand back 60 cards from another deck.
    """
    here = _os.path.dirname(_os.path.abspath(__file__))
    root = _os.path.dirname(_os.path.dirname(here))
    for path in (
        _os.path.join(here, "deck.csv"),
        "/kaggle_simulations/agent/deck.csv",
        _os.path.join(root, "crustle_deck.csv"),
    ):
        if _os.path.exists(path):
            with open(path) as f:
                cards = [int(line) for line in f.readlines() if line.strip()]
            if len(cards) >= 60:
                return cards[:60]
    raise FileNotFoundError("crustle_agent: could not locate a 60-card deck.csv")


deck: list[int] = _read_deck()


def set_deck(deck_list: list[int]) -> None:
    """Override the deck loaded from deck.csv. Called by main.py if used."""
    global deck
    deck = deck_list


# ── Card database (optional) ──────────────────────────────────────────────
# Used only to reason about the *opponent's* cards -- HP, prize value,
# Fighting typing. Every consumer degrades to a neutral answer when it is
# unavailable, so the agent still plays a coherent game without it.
try:
    from ptcg.api import all_card_data

    CARD_DB = {c.cardId: c for c in all_card_data()}
except Exception:  # pragma: no cover - engine not importable
    CARD_DB = {}


def _prize_value(card_id) -> int:
    data = CARD_DB.get(card_id)
    if data is None:
        return 1
    if getattr(data, "megaEx", False):
        return 3
    if getattr(data, "ex", False):
        return 2
    return 1


def _is_ex(card_id) -> bool:
    """ex or Mega Evolution ex -- i.e. an attacker Crustle walls completely."""
    data = CARD_DB.get(card_id)
    return bool(data) and (getattr(data, "ex", False) or getattr(data, "megaEx", False))


def _is_fighting(card_id) -> bool:
    data = CARD_DB.get(card_id)
    return bool(data) and getattr(data, "energyType", None) == EnergyType.FIGHTING


def _weakness(card_id):
    data = CARD_DB.get(card_id)
    return getattr(data, "weakness", None) if data else None


# ── Safe accessors ────────────────────────────────────────────────────────
def _get_card(
    obs: Observation, area: AreaType, index: int, player_index: int
) -> Pokemon | Card | None:
    """Pull a Card/Pokemon out of a zone, returning None instead of raising on
    any out-of-range or missing-zone combination."""
    current = obs.current
    if current is None or index is None or area is None:
        return None

    if area == AreaType.DECK:
        deck_cards = obs.select.deck if obs.select else None
        if deck_cards and 0 <= index < len(deck_cards):
            return deck_cards[index]
        return None
    if area == AreaType.STADIUM:
        return current.stadium[index] if 0 <= index < len(current.stadium) else None
    if area == AreaType.LOOKING:
        looking = current.looking
        return looking[index] if looking and 0 <= index < len(looking) else None

    if player_index is None or not (0 <= player_index < len(current.players)):
        return None
    ps: PlayerState = current.players[player_index]

    zone = {
        AreaType.HAND: ps.hand,
        AreaType.DISCARD: ps.discard,
        AreaType.ACTIVE: ps.active,
        AreaType.BENCH: ps.bench,
        AreaType.PRIZE: ps.prize,
    }.get(area)
    if zone is None or not (0 <= index < len(zone)):
        return None
    return zone[index]


def _active(ps: PlayerState) -> Pokemon | None:
    return ps.active[0] if ps and ps.active and ps.active[0] else None


def _card_id(card) -> int | None:
    return getattr(card, "id", None)


def _hand_ids(ps: PlayerState) -> list[int]:
    return [c.id for c in (ps.hand or []) if c is not None]


def _damage_on(mon: Pokemon) -> int:
    if mon is None or mon.hp is None or mon.maxHp is None:
        return 0
    return max(0, mon.maxHp - mon.hp)


def _has_grass(mon: Pokemon) -> bool:
    """Does this Pokemon have a {G} source attached? RAINBOW counts as any
    type, so it satisfies the {G} half of Superb Scissors too."""
    return any(
        e in (EnergyType.GRASS, EnergyType.RAINBOW) for e in (mon.energies or [])
    )


def _can_attack(mon: Pokemon) -> bool:
    """Crustle readiness: 3 Energy *including a {G} source*. See the module
    docstring -- three colourless Energy is a brick, not an attacker."""
    if mon is None or mon.id != CRUSTLE:
        return False
    return len(mon.energies or []) >= CRUSTLE_ATTACK_COST and _has_grass(mon)


def _damage_vs(attacker_damage: int, defender: Pokemon, attack_type) -> int:
    """Weakness-adjusted damage. Resistance is ignored -- it is a flat -20 or
    -30 and never flips a KO check that Weakness didn't already decide."""
    if defender is None:
        return attacker_damage
    if _weakness(defender.id) == attack_type:
        return attacker_damage * 2
    return attacker_damage


def _crustle_kos(defender: Pokemon) -> bool:
    if defender is None or defender.hp is None:
        return False
    return _damage_vs(CRUSTLE_ATTACK_DAMAGE, defender, EnergyType.GRASS) >= defender.hp


# ── Board view ────────────────────────────────────────────────────────────
class _Board:
    """Everything the scorers need, computed once per agent() call."""

    def __init__(self, obs: Observation, current: State, your_index: int):
        self.obs = obs
        self.current = current
        self.me = your_index
        self.opp = 1 - your_index
        self.ps: PlayerState = current.players[your_index]
        self.ops: PlayerState = current.players[self.opp]

        self.active = _active(self.ps)
        self.active_id = _card_id(self.active)
        self.opp_active = _active(self.ops)
        self.hand_ids = _hand_ids(self.ps)

        self.stadium_id = current.stadium[0].id if current.stadium else None
        self.stadium_owner = (
            current.stadium[0].playerIndex if current.stadium else None
        )
        self.factory = self.stadium_id == TEAM_ROCKETS_FACTORY

        # (area, index, mon) for every Dwebble/Crustle we control.
        self.line: list[tuple[AreaType, int, Pokemon]] = []
        if self.active is not None and self.active_id in CRUSTLE_LINE_IDS:
            self.line.append((AreaType.ACTIVE, 0, self.active))
        for i, mon in enumerate(self.ps.bench or []):
            if mon is not None and mon.id in CRUSTLE_LINE_IDS:
                self.line.append((AreaType.BENCH, i, mon))

        self.line_count = len(self.line)
        self.kangaskhan_count = sum(
            1
            for mon in ([self.active] if self.active else []) + list(self.ps.bench or [])
            if mon is not None and mon.id == MEGA_KANGASKHAN_EX
        )

        self.active_ready = _can_attack(self.active)
        self.ready_bench = [
            (area, idx, mon)
            for area, idx, mon in self.line
            if area == AreaType.BENCH and _can_attack(mon)
        ]
        # The wall is built but stuck behind something else -- Switch time.
        self.stuck = bool(self.ready_bench) and not self.active_ready

        self.build_target = self._pick_build_target()

        # Opponent read.
        self.opp_board = [m for m in ([self.opp_active] + list(self.ops.bench or [])) if m]
        self.opp_bench = [m for m in (self.ops.bench or []) if m]
        self.opp_has_fighting = any(_is_fighting(m.id) for m in self.opp_board)
        self.opp_froslass = any(m.id in FROSLASS_LINE_IDS for m in self.opp_board)
        # Battle Cage stops damage counters being placed on BENCHED Pokemon by
        # the opponent's attack effects and Abilities. Freezing Shroud is the
        # named case, but Phantom Dive-style bench sniping is the same shape,
        # so the trigger is the observable symptom rather than a card list:
        # our Bench is taking damage at all. Nothing in a normal exchange
        # damages the Bench, so this is close to a direct read of "an effect
        # is chipping the pieces I am building".
        self.bench_taking_damage = any(
            _damage_on(m) > 0 for m in (self.ps.bench or []) if m is not None
        )
        self.wants_battle_cage = (
            any(m.id == FROSLASS_TWM for m in self.opp_board) or self.bench_taking_damage
        )
        self.opp_has_energy = any(len(m.energies or []) > 0 for m in self.opp_board)
        self.opp_hand_count = self.ops.handCount or 0
        # Non-ex attackers are the only opposing Pokemon that can damage the
        # wall at all, so they are what Boss's Orders wants gone.
        self.opp_threats = [m for m in self.opp_board if not _is_ex(m.id)]

    def _pick_build_target(self):
        """Which line member the once-per-turn Energy should go on.

        Crustle before Dwebble, then the one closest to attacking, then the
        Active (it can attack the soonest). Returns None when every line
        member is already able to attack, or when there is no line in play.
        """
        candidates = [
            (area, idx, mon) for area, idx, mon in self.line if not _can_attack(mon)
        ]
        if not candidates:
            # Every line member can already attack (so every one of them has
            # its {G}). Topping one up is still better than feeding
            # Kangaskhan, but only up to MAX_USEFUL_ENERGY.
            candidates = [
                (area, idx, mon)
                for area, idx, mon in self.line
                if len(mon.energies or []) < MAX_USEFUL_ENERGY
            ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda t: (
                t[2].id == CRUSTLE,
                len(t[2].energies or []),
                t[0] == AreaType.ACTIVE,
            ),
        )

    def is_build_target(self, area, index) -> bool:
        if self.build_target is None:
            return False
        return self.build_target[0] == area and self.build_target[1] == index

    def in_play(self, area, index) -> Pokemon | None:
        if area == AreaType.ACTIVE:
            return self.active if index in (0, None) else None
        if area == AreaType.BENCH:
            bench = self.ps.bench or []
            return bench[index] if index is not None and 0 <= index < len(bench) else None
        return None

    def wants_ice_cream(self) -> bool:
        """Active is hurt enough for Jumbo Ice Cream to be worth a card, and
        meets its 3-Energy requirement."""
        return (
            self.active is not None
            and len(self.active.energies or []) >= 3
            and _damage_on(self.active) >= 40
        )

    def needs_crustle_in_hand(self) -> bool:
        """A Dwebble is in play with nothing to evolve it into."""
        return any(mon.id == DWEBBLE for _, _, mon in self.line) and (
            CRUSTLE not in self.hand_ids
        )

    def cape_in_play(self) -> bool:
        return any(
            any(t.id == HEROS_CAPE for t in (mon.tools or []))
            for _, _, mon in self.line
        )

    def best_boss_target(self) -> tuple[Pokemon | None, int]:
        """(target, value) for the best Boss's Orders drag off their bench."""
        best, best_value = None, 0
        for mon in self.opp_bench:
            value = self.boss_value(mon)
            if value > best_value:
                best, best_value = mon, value
        return best, best_value

    def boss_value(self, mon: Pokemon) -> int:
        """How much we want this Pokemon dragged into the Active Spot."""
        if mon is None:
            return 0
        value = 0
        if mon.id == FROSLASS_TWM:
            # The one card that ignores Mysterious Rock Inn. Kill it on sight.
            value += 2000
        elif mon.id in FROSLASS_PRE_EVOLUTION_IDS:
            # A Snorunt is only *maybe* the threat: it evolves into either
            # Froslass (Freezing Shroud) or Mega Froslass ex (no Ability that
            # touches us). Killing it early is nice, but not worth overriding
            # the rest of the table on a coin-flip read -- which is what the
            # old flat +2000 on the whole line did, and the starmie deck
            # (Snorunt -> Mega Froslass ex) is exactly the case it got wrong.
            value += 400
        if _crustle_kos(mon):
            value += 600 + 200 * _prize_value(mon.id)
        if not _is_ex(mon.id):
            # Non-ex bodies are the only ones that can damage the wall.
            value += 300
        value += 25 * len(mon.energies or [])
        return value


def _hand_trimmer_gain(b: _Board) -> int:
    """Net cards Hand Trimmer strips from the opponent, minus what it costs us.

    "Each player discards down to 5, your opponent discards first." Playing it
    takes it out of our own hand before either player counts, hence the -1 on
    our side. Symmetric, so it is only worth a card when our hand is already
    lean and theirs is not -- which is exactly the board this deck grinds
    toward, since Lillie empties our hand every few turns.
    """
    theirs = max(0, b.opp_hand_count - 5)
    ours = max(0, (len(b.hand_ids) - 1) - 5)
    return theirs - ours


# ── How badly we want each card ───────────────────────────────────────────
def _want(card_id: int, b: _Board) -> int:
    """Value of adding this card to hand right now.

    Drives every search (Petrel, Hilda, Ultra Ball, Pokegear, Poffin) and,
    inverted, every discard. Situational so the same card scores differently
    early and late.
    """
    if card_id == CRUSTLE:
        return 1000 if b.needs_crustle_in_hand() else 600
    if card_id == DWEBBLE:
        return 800 if b.line_count == 0 else (450 if b.line_count < 2 else 100)
    if card_id == SWITCH:
        return 1100 if b.stuck else 250
    if card_id == JUMBO_ICE_CREAM:
        return 900 if b.wants_ice_cream() else 300
    if card_id == HEROS_CAPE:
        return 700 if (b.line_count > 0 and not b.cape_in_play()) else 200
    if card_id == GROW_GRASS_ENERGY:
        # The {G} half of Superb Scissors, and only 5 sources in 60 cards.
        if b.build_target is not None and not _has_grass(b.build_target[2]):
            return 850
        return 400
    if card_id == BASIC_GRASS_ENERGY:
        if b.build_target is not None and not _has_grass(b.build_target[2]):
            return 700
        return 200
    if card_id == MIST_ENERGY:
        return 450
    if card_id == SPIKY_ENERGY:
        return 250
    if card_id == BOSSS_ORDERS:
        return 700 if b.best_boss_target()[1] >= 600 else 300
    if card_id == BUDDY_BUDDY_POFFIN:
        return 750 if b.line_count < MAX_CRUSTLE_LINE_IN_PLAY else 120
    if card_id == HILDA:
        return 700 if b.needs_crustle_in_hand() else 350
    if card_id == TEAM_ROCKETS_PETREL:
        return 500 + (100 if b.factory else 0)
    if card_id == ULTRA_BALL:
        return 550 if b.line_count == 0 else 200
    if card_id == POKEGEAR_3:
        return 350
    if card_id == CRUSHING_HAMMER:
        return 350 if b.opp_has_energy else 150
    if card_id == XEROSICS_MACHINATIONS:
        return 500 if b.opp_hand_count >= 6 else 150
    if card_id == HAND_TRIMMER:
        return 450 if _hand_trimmer_gain(b) >= 2 else 150
    if card_id == LILLIES_DETERMINATION:
        return 400 if len(b.hand_ids) <= 3 else 150
    if card_id == BATTLE_CAGE:
        # The only card in the 60 that answers Freezing Shroud, so it is worth
        # digging for specifically rather than as a generic stadium.
        if b.wants_battle_cage and b.stadium_id != BATTLE_CAGE:
            return 900
        return 300 if b.stadium_owner == b.opp else 180
    if card_id in STADIUM_IDS:
        return 300 if b.stadium_owner == b.opp else 180
    if card_id == MEGA_KANGASKHAN_EX:
        # Exactly one is the plan; a second is 3 more prizes on the table.
        return 600 if b.kangaskhan_count == 0 else 60
    return 200


# ── MAIN-turn scoring ─────────────────────────────────────────────────────
def _score_ability(o, b: _Board) -> int:
    card = _get_card(b.obs, o.area, o.index, b.me)
    if _card_id(card) == MEGA_KANGASKHAN_EX:
        # Run Errand: free draw 2. Always first -- more cards means better
        # choices for every action below. Held back only when the extra draw
        # would eat the last of the deck.
        return 400 if (b.ps.deckCount or 0) <= 2 else 3000
    return 900


def _score_attach(o, b: _Board) -> int:
    card = _get_card(b.obs, o.area, o.index, b.me)
    cid = _card_id(card)
    target = b.in_play(o.inPlayArea, o.inPlayIndex)
    if target is None:
        return -10

    if cid == HEROS_CAPE:
        # +100 HP. Tools survive evolution, so a Dwebble that is about to
        # become Crustle is a fine home for it.
        if any(t.id == HEROS_CAPE for t in (target.tools or [])):
            return -10
        if target.id == CRUSTLE:
            return 2450
        if target.id == DWEBBLE:
            return 2200
        # Dead weight on Kangaskhan unless the line never showed up.
        return 700 if b.line_count == 0 else -10

    if cid not in ENERGY_IDS:
        return 900  # unknown tool: play it somewhere sane

    # The Energy cap must never block the {G} source Superb Scissors needs.
    # A line member holding four colourless Energy is a brick, and suppressing
    # the Grow Grass attach onto it would strand it permanently.
    fixes_grass = (
        target.id in CRUSTLE_LINE_IDS
        and not _has_grass(target)
        and cid in GRASS_SOURCE_IDS
    )
    if len(target.energies or []) >= MAX_USEFUL_ENERGY and not fixes_grass:
        return -10

    # Energy card preference. Cover the {G} requirement first, then Mist
    # (which plugs the effect-damage hole in Mysterious Rock Inn), then Spiky.
    if target.id in CRUSTLE_LINE_IDS and not _has_grass(target):
        card_bonus = {
            GROW_GRASS_ENERGY: 120,
            BASIC_GRASS_ENERGY: 100,
            MIST_ENERGY: 20,
            SPIKY_ENERGY: 10,
        }.get(cid, 0)
    else:
        card_bonus = {
            MIST_ENERGY: 100,
            GROW_GRASS_ENERGY: 80,
            SPIKY_ENERGY: 50,
            BASIC_GRASS_ENERGY: 20,
        }.get(cid, 0)

    if b.is_build_target(o.inPlayArea, o.inPlayIndex):
        return 2100 + card_bonus
    if target.id in CRUSTLE_LINE_IDS:
        return 1400 + card_bonus
    if target.id == MEGA_KANGASKHAN_EX:
        # Only when no line member wants it -- otherwise this is the single
        # worst attachment in the deck.
        return -10 if b.build_target is not None else 900 + card_bonus
    return 800 + card_bonus


def _score_evolve(o, b: _Board) -> int:
    card = _get_card(b.obs, o.area, o.index, b.me)
    if _card_id(card) == CRUSTLE:
        target = b.in_play(o.inPlayArea, o.inPlayIndex)
        # Prefer evolving the Dwebble that already carries the Energy.
        bonus = 20 * len(target.energies or []) if target else 0
        return 2600 + bonus
    return 800


def _score_play(o, b: _Board) -> int:
    card = _get_card(b.obs, AreaType.HAND, o.index, b.me)
    cid = _card_id(card)
    if cid is None:
        return 700

    # ── Pokemon ───────────────────────────────────────────────────────────
    if cid == DWEBBLE:
        return 1800 if b.line_count < MAX_CRUSTLE_LINE_IN_PLAY else -10
    if cid == MEGA_KANGASKHAN_EX:
        if b.kangaskhan_count >= MAX_KANGASKHAN_IN_PLAY:
            return -10
        # 300 HP halves against Fighting and hands over 3 prizes.
        return 250 if b.opp_has_fighting else 1500

    # ── Swap / heal ───────────────────────────────────────────────────────
    if cid == SWITCH:
        # Retreat costs 3 Energy for both walls, so this is the real swap.
        return 2300 if b.stuck else -10
    if cid == JUMBO_ICE_CREAM:
        if b.active is None or len(b.active.energies or []) < 3:
            return -10
        damage = _damage_on(b.active)
        if damage >= 70:
            return 1900          # at or near the full 80
        if damage >= 40:
            return 950
        return -10               # 4 copies; don't burn one to heal 20
    # ── Targeting ─────────────────────────────────────────────────────────
    if cid == BOSSS_ORDERS:
        if not b.opp_bench:
            return -10
        _, value = b.best_boss_target()
        if value >= 2000:
            return 2500          # Froslass -- the deck's only real out
        # Only worth the turn's Supporter if something can convert it.
        can_convert = b.active_ready or b.stuck
        if value >= 600:
            return 2200 if can_convert else 1000
        return 300

    # ── Search ────────────────────────────────────────────────────────────
    if cid == TEAM_ROCKETS_PETREL:
        # Searches any Trainer, so it is whichever piece we are missing.
        score = 1200 + (150 if b.factory else 0)
        if b.stuck and SWITCH not in b.hand_ids:
            score = 2250         # go get the Switch
        elif b.wants_ice_cream() and JUMBO_ICE_CREAM not in b.hand_ids:
            score = 1900
        elif b.line_count == 0:
            score = 1700         # fetch Poffin / Ultra Ball to find the line
        return score
    if cid == HILDA:
        # Evolution Pokemon + an Energy card: Crustle plus Grow Grass.
        return 2000 if b.needs_crustle_in_hand() else 1100
    if cid == BUDDY_BUDDY_POFFIN:
        # Dwebble is 70 HP so it is a legal target; Kangaskhan at 300 is not.
        return 2050 if b.line_count < MAX_CRUSTLE_LINE_IN_PLAY else -10
    if cid == ULTRA_BALL:
        # Costs 2 cards -- only when the board actually needs a body.
        if b.line_count == 0 or b.needs_crustle_in_hand():
            return 1500
        return -10 if len(b.hand_ids) <= 3 else 400
    if cid == POKEGEAR_3:
        supporters_in_hand = any(c in DECK_SUPPORTER_IDS for c in b.hand_ids)
        return 800 if supporters_in_hand else 1450

    # ── Draw / disruption ─────────────────────────────────────────────────
    if cid == LILLIES_DETERMINATION:
        # Shuffles the hand away for 6 (8 at exactly 6 Prizes), so its cost is
        # everything it throws away. hand_ids still contains Lillie itself.
        discarded = max(0, len(b.hand_ids) - 1)
        if discarded <= 1:
            return 2000
        if discarded <= 3:
            return 1250
        if discarded <= 5:
            return 500
        return -10
    if cid == XEROSICS_MACHINATIONS:
        if b.opp_hand_count >= 7:
            return 1600
        if b.opp_hand_count >= 5:
            return 1000
        return -10
    if cid == HAND_TRIMMER:
        # Symmetric, and it costs a card, so it needs a real edge to be worth
        # playing. Best right after a Lillie has already emptied our hand.
        gain = _hand_trimmer_gain(b)
        if gain >= 4:
            return 1500
        if gain >= 2:
            return 1000
        return -10
    if cid == CRUSHING_HAMMER:
        return 1050 if b.opp_has_energy else -10

    # ── Stadiums ──────────────────────────────────────────────────────────
    if cid in STADIUM_IDS:
        if b.stadium_owner == b.me:
            # Ours is already down. Replacing it burns a card to swap one
            # effect for another, so the only case worth it is upgrading into
            # Battle Cage when something is actually chipping our Bench. The
            # old check only blocked replacing a stadium with *itself*, which
            # with four different stadiums in the deck let the agent cycle its
            # own Community Center out for no gain.
            upgrading = (
                cid == BATTLE_CAGE
                and b.wants_battle_cage
                and b.stadium_id != BATTLE_CAGE
            )
            return 1700 if upgrading else -10
        # Bumping the opponent's stadium is worth more than playing into open
        # space, since theirs is presumably doing something for them.
        score = 1600 if (b.stadium_id is not None and b.stadium_owner == b.opp) else 1000
        if cid == COMMUNITY_CENTER and any(
            _damage_on(m) > 0
            for m in ([b.active] if b.active else []) + list(b.ps.bench or [])
            if m
        ):
            score += 250         # heal 10 across a 250+ HP board adds up
        if cid == TEAM_ROCKETS_FACTORY and TEAM_ROCKETS_PETREL in b.hand_ids:
            score += 200         # Petrel has "Team Rocket" in its name
        if cid == FESTIVAL_GROUNDS and (
            b.ps.poisoned or b.ps.burned or b.ps.asleep
            or b.ps.paralyzed or b.ps.confused
        ):
            score += 400         # status is the other hole in the wall
        if cid == BATTLE_CAGE and b.wants_battle_cage:
            # Freezing Shroud is an Ability placing damage counters, so Battle
            # Cage blanks it for everything on our Bench -- the single best
            # answer in the 60 to the line that ignores Mysterious Rock Inn.
            # It does NOT cover the Active Spot, which is why Boss'ing the
            # Froslass and killing it stays the higher priority.
            score += 900
        return score

    return 700


def _score_retreat(b: _Board) -> int:
    """Retreating discards the retreat cost in Energy (3 for both walls), so
    it is normally strictly worse than Switch. The exception is Kangaskhan:
    Energy on it is dead weight anyway, so paying with it to get the built
    Crustle in front is fine."""
    if b.active is None:
        return -3
    if b.active_id == CRUSTLE:
        return -5                # never throw away the investment
    if b.stuck and SWITCH not in b.hand_ids:
        if b.active_id == MEGA_KANGASKHAN_EX:
            return 1300
        return 1100              # Dwebble, retreat cost 2
    return -3


def _score_attack(o, b: _Board) -> int:
    """Attacks end the turn, so the whole band sits under every setup action."""
    attack_id = o.attackId

    if attack_id == SUPERB_SCISSORS:
        return 500 if _crustle_kos(b.opp_active) else 400

    if attack_id == ASCENSION:
        # 0 damage, but it evolves straight out of the deck -- by far the best
        # thing a Dwebble stuck in the Active Spot can be doing.
        if CRUSTLE in b.hand_ids:
            return 200           # a manual evolve was free; something blocked it
        return 450 if b.line_count <= MAX_CRUSTLE_LINE_IN_PLAY else 200

    if attack_id == RAPID_FIRE_COMBO:
        score = 350 if (
            b.opp_active is not None
            and b.opp_active.hp is not None
            and _damage_vs(KANGASKHAN_ATTACK_DAMAGE, b.opp_active, EnergyType.COLORLESS)
            >= b.opp_active.hp
        ) else 250
        if b.ready_bench:
            # We should have swapped instead of exposing the 3-prize body.
            score -= 120
        return score

    return 300


def _score_main(o, b: _Board) -> int:
    if o.type == OptionType.ABILITY:
        return _score_ability(o, b)
    if o.type == OptionType.ATTACH:
        return _score_attach(o, b)
    if o.type == OptionType.EVOLVE:
        return _score_evolve(o, b)
    if o.type == OptionType.PLAY:
        return _score_play(o, b)
    if o.type == OptionType.RETREAT:
        return _score_retreat(b)
    if o.type == OptionType.ATTACK:
        return _score_attack(o, b)
    if o.type == OptionType.DISCARD:
        return -10               # discarding our own in-play card is never the plan
    if o.type == OptionType.END:
        return 100
    return 0


# ── Sub-selection scoring ─────────────────────────────────────────────────
# Everything here starts from a solid positive base so the agent always makes
# a *valid* choice and the game keeps moving (the notebook's rule), then adds
# deck-specific preferences on top.
_SUB_BASE = 2000

# Contexts where the selected card is being taken away from its owner. When
# the options are ours we pick the fewest and cheapest; when they are the
# opponent's we pick the most and most expensive.
_COST_CONTEXTS = frozenset(
    {
        SelectContext.DISCARD,
        SelectContext.DISCARD_CARD_OR_ATTACHED_CARD,
        SelectContext.DISCARD_ENERGY,
        SelectContext.DISCARD_ENERGY_CARD,
        SelectContext.DISCARD_TOOL_CARD,
        SelectContext.TO_DECK,
        SelectContext.TO_DECK_BOTTOM,
        SelectContext.TO_PRIZE,
        SelectContext.DEVOLVE,
    }
)

_SEARCH_CONTEXTS = frozenset(
    {
        SelectContext.TO_HAND,
        SelectContext.TO_FIELD,
        SelectContext.LOOK,
        SelectContext.EVOLVES_TO,
    }
)


def _energy_card_at(b: _Board, o) -> int | None:
    """Card id behind an ENERGY / ENERGY_CARD option (they address a Pokemon
    by area+index and then an energyIndex into its attached Energy)."""
    if o.playerIndex is not None and o.playerIndex != b.me:
        return None
    mon = b.in_play(o.area, o.index)
    if mon is None or o.energyIndex is None:
        return None
    cards = mon.energyCards or []
    return cards[o.energyIndex].id if 0 <= o.energyIndex < len(cards) else None


def _mon_at(current: State, player_index, area, index) -> Pokemon | None:
    """A Pokemon in play on either side, addressed the way options address it."""
    if player_index is None or not (0 <= player_index < len(current.players)):
        return None
    ps = current.players[player_index]
    if area == AreaType.ACTIVE:
        zone = ps.active
    elif area == AreaType.BENCH:
        zone = ps.bench
    else:
        return None
    if index is None or not (0 <= index < len(zone or [])):
        return None
    return zone[index]


def _score_sub(o, b: _Board | None, context, sel: SelectData, obs: Observation) -> int:
    score = _SUB_BASE

    # ── Yes/No ────────────────────────────────────────────────────────────
    if o.type == OptionType.YES:
        if context == SelectContext.IS_FIRST:
            # A wall deck wants the extra setup turn far more than it wants
            # the turn-1 attack it cannot make anyway.
            return score + 500
        return score + 100
    if o.type == OptionType.NO:
        return score
    if o.type == OptionType.NUMBER:
        return score + (o.number or 0)
    if o.type == OptionType.SPECIAL_CONDITION:
        return score

    if b is None:
        return score

    # ── Attached Energy (retreat cost, Crushing Hammer, ...) ──────────────
    if o.type in (OptionType.ENERGY, OptionType.ENERGY_CARD):
        if o.playerIndex is not None and o.playerIndex != b.me:
            # Their Energy: strip from whoever is most invested, Active first.
            mon = _mon_at(b.current, o.playerIndex, o.area, o.index)
            score += 300
            score += 40 * len(mon.energies or []) if mon is not None else 0
            score += 200 if o.area == AreaType.ACTIVE else 0
            return score
        cid = _energy_card_at(b, o)
        # Ours, and about to be discarded: shed the least useful first, and
        # hang on to the {G} sources Superb Scissors needs.
        keep = {
            GROW_GRASS_ENERGY: 400,
            BASIC_GRASS_ENERGY: 350,
            MIST_ENERGY: 250,
            SPIKY_ENERGY: 120,
        }.get(cid, 200)
        return score + (500 - keep)
    if o.type == OptionType.TOOL_CARD:
        return score
    if o.type == OptionType.SKILL:
        return score

    if o.type != OptionType.CARD:
        return score

    owner = o.playerIndex if o.playerIndex is not None else b.me
    card = _get_card(obs, o.area, o.index, owner)
    if card is None:
        return score
    cid = card.id
    mine = owner == b.me
    effect_id = _card_id(sel.effect)

    # ── Cards being taken away ────────────────────────────────────────────
    if context in _COST_CONTEXTS:
        if not mine:
            # Their cards, being taken away: hit their best stuff.
            return score + 400 + _want(cid, b)
        # Ultra Ball's cost, hand-size trims: invert the want table so the
        # least useful card is the one that goes.
        return score + max(0, 1100 - _want(cid, b))

    # ── Opening board ─────────────────────────────────────────────────────
    if context == SelectContext.SETUP_ACTIVE_POKEMON:
        # Kangaskhan leads: a 300 HP body that draws 2 a turn while the bench
        # develops. Leading a Dwebble puts a 70 HP starter in the firing line.
        return score + (900 if cid == MEGA_KANGASKHAN_EX else 100)
    if context in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
        if cid == DWEBBLE:
            return score + (900 if b.line_count < MAX_CRUSTLE_LINE_IN_PLAY else 150)
        if cid == MEGA_KANGASKHAN_EX:
            return score + (400 if b.kangaskhan_count == 0 else 50)
        return score + 200

    # ── Bringing something to the Active Spot ─────────────────────────────
    if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
        mon = card if isinstance(card, Pokemon) else None
        if mon is not None and _can_attack(mon):
            return score + 1200
        if cid == CRUSTLE:
            return score + 700
        if cid == MEGA_KANGASKHAN_EX:
            return score + 400
        if cid == DWEBBLE:
            return score + 100
        return score + 200

    # ── Searches ──────────────────────────────────────────────────────────
    if context in _SEARCH_CONTEXTS:
        return score + _want(cid, b)
    if context == SelectContext.EVOLVES_FROM:
        # Evolve the Dwebble carrying the most Energy.
        mon = card if isinstance(card, Pokemon) else None
        return score + 300 + 30 * (len(mon.energies or []) if mon else 0)
    if context == SelectContext.EVOLVE:
        return score + _want(cid, b)

    # ── Healing ourselves ─────────────────────────────────────────────────
    if context in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
        mon = card if isinstance(card, Pokemon) else None
        if mon is None:
            return score
        bonus = _damage_on(mon)
        if mon.id in CRUSTLE_LINE_IDS:
            bonus += 300
        return score + bonus

    # ── Pointing an effect at something ───────────────────────────────────
    if context in (
        SelectContext.EFFECT_TARGET,
        SelectContext.DAMAGE,
        SelectContext.DAMAGE_COUNTER,
        SelectContext.DAMAGE_COUNTER_ANY,
        SelectContext.ATTACH_FROM,
        SelectContext.ATTACH_TO,
        SelectContext.DETACH_FROM,
    ):
        mon = card if isinstance(card, Pokemon) else None

        if effect_id == BOSSS_ORDERS and mon is not None:
            return score + b.boss_value(mon)
        if effect_id == CRUSHING_HAMMER and mon is not None and not mine:
            # Strip from whoever is most invested; their Active first.
            return score + 200 * len(mon.energies or []) + (
                200 if o.area == AreaType.ACTIVE else 0
            )
        if effect_id == HEROS_CAPE and mon is not None and mine:
            if mon.id == CRUSTLE:
                return score + 800
            if mon.id == DWEBBLE:
                return score + 500
            return score + 50
        if effect_id in ENERGY_IDS and mon is not None and mine:
            if b.is_build_target(o.area, o.index):
                return score + 900
            if mon.id in CRUSTLE_LINE_IDS:
                return score + 500
            return score + 50

        if mon is not None and not mine:
            # Generic hostile targeting, same shape as the notebook's rule:
            # their Active first, then whoever is most invested.
            return score + (500 if o.area == AreaType.ACTIVE else 100) + 40 * len(
                mon.energies or []
            )
        if mon is not None:
            return score + (mon.hp or 0)
        return score + _want(cid, b)

    # ── Fallback: the notebook's generic rule ─────────────────────────────
    if isinstance(card, Pokemon):
        if not mine:
            return score + (500 if o.area == AreaType.ACTIVE else 100) + 50 * len(
                card.energies or []
            )
        return score + (card.hp or 0)
    return score + _want(cid, b)


def _desired_count(context, options, b: _Board | None, min_count, max_count) -> int:
    """How many options to actually return.

    'Up to N' selections are only free when the thing being selected is good
    for us. When the engine is asking which of *our* cards to throw away we
    take the minimum it will accept -- the notebook's version fills to
    maxCount whenever the scores are positive, which over-discards.

    It checks ownership rather than just the context because an effect that
    discards from the *opponent's* side arrives as the same DISCARD select,
    and there we do want the maximum. (The 60 no longer runs Eri, which is
    what motivated this; Hand Trimmer's own discard is ours, so it correctly
    takes the minimum.)
    """
    if context in _COST_CONTEXTS and b is not None:
        hits_opponent = any(o.playerIndex == b.opp for o in options)
        if not hits_opponent:
            return min_count
    return max_count


def agent(obs_dict: dict) -> list[int]:
    """Score every option, return the best ones.

    Priority order, highest first:
        3000  Run Errand (Kangaskhan's free draw 2)
        2600  Evolve Dwebble -> Crustle
        2500  Boss's Orders onto the Froslass line
        2450  Hero's Cape onto the Crustle line
        2300  Switch to a benched Crustle that can attack
        2250  Petrel to fetch that Switch
        2200  Boss's Orders with a convertible target
        2100  Energy onto the build target ({G} source first)
        2050  Buddy-Buddy Poffin / 2000 Hilda / 2000 Lillie on an empty hand
        1900  Battle Cage vs a Froslass board / Jumbo Ice Cream at full value
        1800  Dwebble to the bench / 1600 stadiums / 1500 Kangaskhan
        1500  Hand Trimmer when it strips 4+ more from them than from us
        1050  Crushing Hammer while they have Energy
         700  anything else playable
         300-500  attacks (they end the turn)
         100  END
        <  0  suppressed unless minCount forces it
    """
    if obs_dict.get("select") is None:
        return deck

    obs: Observation = to_observation_class(obs_dict)
    sel: SelectData = obs.select
    options = sel.option
    if not options:
        return []

    context: SelectContext = sel.context
    current: State | None = obs.current
    your_index = current.yourIndex if current is not None else None

    max_count = min(sel.maxCount, len(options))
    min_count = min(sel.minCount, max_count)

    board: _Board | None = None
    if current is not None and your_index is not None and len(current.players) == 2:
        try:
            board = _Board(obs, current, your_index)
        except Exception:
            board = None

    scores: list[int] = []
    for o in options:
        if context == SelectContext.MAIN and board is not None:
            scores.append(_score_main(o, board))
        else:
            scores.append(_score_sub(o, board, context, sel, obs))

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    want = _desired_count(context, options, board, min_count, max_count)

    output: list[int] = []
    for i in order:
        if len(output) >= want and len(output) >= min_count:
            break
        # Negative scores are suppressed actions -- take them only when the
        # engine will not accept a shorter answer.
        if scores[i] >= 0 or len(output) < min_count:
            output.append(i)

    return output
