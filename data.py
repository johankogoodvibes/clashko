"""Load collected Ladder matches and encode card and tower-troop features."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "combined_dataset" / "recent_matches.csv"
# CSV_FILE = DATA_DIR / "player_check" / "recent_matches.csv"
MAX_LEVEL = 16
MAX_ELIXIR_LEAKED = None # 9.0  # Both players must be below this value. None disables the filter.
MIN_TROPHIES = 13_200  # Both players must meet this minimum. None disables the filter.
MAX_TROPHIES = 14_500  # Both players must be at or below this maximum.
MAX_ROWS = None # Cap eligible matches after filtering. None keeps them all.

# Fixed card order shared by both decks; preserve this order for saved models.
card_names = [
    "Archer Queen", "Archers", "Arrows", "Baby Dragon", "Balloon", "Bandit",
    "Barbarian Barrel", "Barbarian Hut", "Barbarians", "Bats", "Battle Healer", "Battle Ram",
    "Berserker", "Bomb Tower", "Bomber", "Boss Bandit", "Bowler", "Cannon",
    "Cannon Cart", "Clone", "Dark Prince", "Dart Goblin", "Earthquake", "Electro Dragon",
    "Electro Giant", "Electro Spirit", "Electro Wizard", "Elite Barbarians", "Elixir Collector", "Elixir Golem",
    "Executioner", "Fire Spirit", "Fireball", "Firecracker", "Fisherman", "Flying Machine",
    "Freeze", "Furnace", "Giant", "Giant Skeleton", "Giant Snowball", "Goblin Barrel",
    "Goblin Cage", "Goblin Curse", "Goblin Demolisher", "Goblin Drill", "Goblin Gang", "Goblin Giant",
    "Goblin Hut", "Goblin Machine", "Goblins", "Goblinstein", "Golden Knight", "Golem",
    "Graveyard", "Guards", "Heal Spirit", "Hog Rider", "Hunter", "Ice Golem",
    "Ice Spirit", "Ice Wizard", "Inferno Dragon", "Inferno Tower", "Knight", "Lava Hound",
    "Lightning", "Little Prince", "Lumberjack", "Magic Archer", "Mega Knight", "Mega Minion",
    "Mighty Miner", "Miner", "Mini P.E.K.K.A", "Minion Giant", "Minion Horde", "Minions",
    "Mirror", "Monk", "Mortar", "Mother Witch", "Musketeer", "Night Witch",
    "P.E.K.K.A", "Phoenix", "Poison", "Prince", "Princess", "Rage",
    "Ram Rider", "Rascals", "Rocket", "Ronin", "Royal Delivery", "Royal Ghost",
    "Royal Giant", "Royal Hogs", "Royal Recruits", "Rune Giant", "Skeleton Army", "Skeleton Barrel",
    "Skeleton Dragons", "Skeleton King", "Skeletons", "Sparky", "Spear Goblins", "Spirit Empress",
    "Suspicious Bush", "Tesla", "The Log", "Three Musketeers", "Tombstone", "Tornado",
    "Valkyrie", "Vines", "Void", "Wall Breakers", "Witch", "Wizard",
    "X-Bow", "Zap", "Zappies",
]

# Tower types have their own fixed rows after the 123 ordinary card rows.
tower_names = ["Tower Princess", "Cannoneer", "Dagger Duchess", "Royal Chef"]
input_names = card_names + tower_names


# Maximum King Tower and Princess Tower HP by level.
# King Tower: Clash Royale Wiki. Tower Troops: crtool game data 16.635.1.
KING_MAX_HP = {
    1: 2400, 2: 2568, 3: 2736, 4: 2904, 5: 3096, 6: 3312, 7: 3528, 8: 3768,
    9: 4008, 10: 4392, 11: 4824, 12: 5304, 13: 5832, 14: 6408, 15: 7032, 16: 7704,
}
TOWER_MAX_HP = {
    "Tower Princess": {
        1: 1400, 2: 1512, 3: 1624, 4: 1750, 5: 1890, 6: 2030, 7: 2184, 8: 2352,
        9: 2534, 10: 2786, 11: 3052, 12: 3346, 13: 3668, 14: 4032, 15: 4424, 16: 4858,
    },
    "Cannoneer": {
        6: 1740, 7: 1872, 8: 2016, 9: 2172, 10: 2388, 11: 2616, 12: 2868, 13: 3144,
        14: 3456, 15: 3792, 16: 4164,
    },
    "Dagger Duchess": {
        9: 2298, 10: 2527, 11: 2768, 12: 3035, 13: 3327, 14: 3657, 15: 4013, 16: 4406,
    },
    "Royal Chef": {9: 2244, 10: 2467, 11: 2703, 12: 2963, 13: 3248, 14: 3571, 15: 3918, 16: 4302},
}


def _post_match_results(battles):
    crown_columns = [f"player_{side}_crowns" for side in (1, 2)]
    crown_counts = battles.reindex(columns=crown_columns).apply(pd.to_numeric, errors="coerce")
    crowns = crown_counts.where(crown_counts.isin([0, 1, 2, 3]), -1).to_numpy(dtype=np.int32)

    hp_columns = [
        f"player_{side}_{tower}_hp"
        for side in (1, 2)
        for tower in ("king_tower", "princess_tower_1", "princess_tower_2")
    ]
    remaining_hp = battles.reindex(columns=hp_columns).apply(pd.to_numeric, errors="coerce")
    hp = remaining_hp.to_numpy(dtype=np.float32, copy=True).reshape(-1, 2, 3)

    for side in range(2):
        side_hp = hp[:, side]
        opposing_crowns = crowns[:, 1 - side]
        # Princess HP is packed in API order. Use the score to fill destroyed slots.
        destroyed = np.column_stack(
            (opposing_crowns == 3, opposing_crowns >= 2, opposing_crowns >= 1)
        )
        side_hp[destroyed & np.isnan(side_hp)] = 0

        levels = pd.to_numeric(battles[f"player_{side + 1}_tower_level"], errors="coerce")
        names = battles[f"player_{side + 1}_tower_name"]
        king_max_hp = levels.map(KING_MAX_HP).to_numpy(dtype=np.float32)
        princess_max_hp = np.array(
            [TOWER_MAX_HP.get(name, {}).get(level, np.nan) for name, level in zip(names, levels)],
            dtype=np.float32,
        )
        maximum_hp = np.column_stack((king_max_hp, princess_max_hp, princess_max_hp))
        hp[:, side] = np.where(side_hp == 0, 0, side_hp / maximum_hp)

    return hp, crowns


def load_data(max_rows=MAX_ROWS, csv_path=None):
    """Return deck inputs, winner labels, normalized tower HP and battle crowns.

    HP has shape (matches, 2, 3) in King, Princess 1, Princess 2 order. The
    Princess slots follow API order, and the King level is assumed to equal
    the equipped Tower Troop level. Unknown HP is NaN. Crowns has shape
    (matches, 2) in player order, with -1 for an unknown score.
    """
    source_path = Path(csv_path) if csv_path is not None else CSV_FILE
    print(f"Loading collected matches from {source_path}", flush=True)
    battles = pd.read_csv(source_path)
    card_columns = [
        f"player_{side}_card_{index}_{field}"
        for side in (1, 2) for index in range(1, 9)
        for field in ("name", "level", "evolution", "hero")
    ]

    battles["battle_time"] = pd.to_datetime(
        battles["battle_time"], format="%Y%m%dT%H%M%S.%fZ", utc=True, errors="coerce"
    )
    battles = battles.loc[battles["game_mode"].eq("Ladder")]
    battles = battles.dropna(subset=card_columns + ["battle_id", "battle_time", "player_1_won"])
    battles = battles.drop_duplicates("battle_id").sort_values(["battle_time", "battle_id"])

    elixir_columns = ["player_1_elixir_leaked", "player_2_elixir_leaked"]
    if MAX_ELIXIR_LEAKED is not None and set(elixir_columns).issubset(battles.columns):
        leaked = battles[elixir_columns].apply(pd.to_numeric, errors="coerce")
        keep = leaked.lt(MAX_ELIXIR_LEAKED).all(axis=1)
        print(f"Leaked-elixir filter: both players < {MAX_ELIXIR_LEAKED:g} | "
              f"kept {keep.sum():,}/{len(battles):,} matches | removed {(~keep).sum():,} "
              f"({leaked.isna().any(axis=1).sum():,} missing/invalid)", flush=True)
        battles = battles.loc[keep]
    elif MAX_ELIXIR_LEAKED is not None:
        print("Leaked-elixir filter skipped: dataset has no complete pair of leaked-elixir columns.", flush=True)

    if MIN_TROPHIES is not None or MAX_TROPHIES is not None:
        players = pd.read_csv(source_path.with_name("players.csv"), usecols=["tag", "mmr"])
        ratings = pd.to_numeric(players.set_index("tag")["mmr"], errors="coerce")
        player_tags = ["player_1_tag", "player_2_tag"]
        trophies = battles[player_tags].apply(lambda tags: tags.map(ratings))
        known = np.isfinite(trophies).all(axis=1)
        keep = known.copy()
        bounds = []
        if MIN_TROPHIES is not None:
            keep &= trophies.ge(MIN_TROPHIES).all(axis=1)
            bounds.append(f">= {MIN_TROPHIES:g}")
        if MAX_TROPHIES is not None:
            keep &= trophies.le(MAX_TROPHIES).all(axis=1)
            bounds.append(f"<= {MAX_TROPHIES:g}")
        print(f"Trophy filter: both players {' and '.join(bounds)} | "
              f"kept {keep.sum():,}/{len(battles):,} matches "
              f"({(~known).sum():,} with missing/invalid ratings)", flush=True)
        battles = battles.loc[keep]

    if max_rows is not None:
        if max_rows < 0:
            raise ValueError("max_rows must be nonnegative or None.")
        battles = battles.head(max_rows)

    name_columns = [f"player_{side}_card_{index}_name" for side in (1, 2) for index in range(1, 9)]
    names = battles[name_columns].to_numpy()
    y = battles["player_1_won"].to_numpy(dtype=np.int32)
    card_indices = pd.Categorical(names.ravel(), categories=card_names).codes
    if (card_indices < 0).any():
        unknown = np.unique(names.ravel()[card_indices < 0])
        raise ValueError(f"Card names missing from the catalog: {unknown.tolist()}")

    card_indices = card_indices.reshape(-1, 2, 8)
    # Features: presence, evolution, hero, visible in-game level / 16.
    X = np.zeros((len(y), 2, len(input_names), 4), dtype=np.float16)
    match_indices = np.arange(len(y))[:, None]
    for side in range(2):
        X[match_indices, side, card_indices[:, side], 0] = 1
        for feature, field in enumerate(("evolution", "hero", "level"), 1):
            columns = [f"player_{side + 1}_card_{index}_{field}" for index in range(1, 9)]
            values = battles[columns].to_numpy(dtype=np.float16)
            if field == "level":
                values = values / MAX_LEVEL
            X[match_indices, side, card_indices[:, side], feature] = values

        towers = battles[f"player_{side + 1}_tower_name"]
        tower_indices = pd.Categorical(towers, categories=tower_names).codes
        unknown_towers = towers.notna().to_numpy() & (tower_indices < 0)
        if unknown_towers.any():
            raise ValueError(f"Tower names missing from the catalog: {towers.loc[unknown_towers].unique().tolist()}")
        present = tower_indices >= 0
        rows = np.arange(len(y))[present]
        indices = len(card_names) + tower_indices[present].astype(np.int32)
        levels = battles[f"player_{side + 1}_tower_level"].fillna(0).to_numpy(dtype=np.float16)
        X[rows, side, indices, 0] = 1
        X[rows, side, indices, 3] = levels[present] / MAX_LEVEL

    hp, crowns = _post_match_results(battles)
    print(
        f"Formatted inputs: {X.shape} | Labels: {y.shape} | "
        f"Tower HP: {hp.shape} | Crowns: {crowns.shape}", flush=True
    )
    return X, y, hp, crowns
