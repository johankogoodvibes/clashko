"""Combine collected MMR folders into one dataset. No API requests."""
from pathlib import Path

from check_players import END_STATE_COLUMNS, MATCH_COLUMNS, save_card_usage
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SOURCE_DIR = Path(__file__).with_name("mmr_bubbles")
OUTPUT_DIR = Path(__file__).with_name("combined_dataset")


def save_plots(folder, rating_title=None):
    players = pd.read_csv(folder / "players.csv")
    matches = pd.read_csv(folder / "recent_matches.csv", usecols=["battle_time"])
    ratings = players["mmr"].dropna()
    ratings.value_counts().sort_index().rename_axis("mmr").rename("players").to_csv(folder / "mmr_distribution.csv")
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.hist(ratings, bins=30)
    ax.set(title=rating_title or f"Observed Ladder ratings — starting category {folder.name}",
           xlabel="Rating (trophies)", ylabel="Players")
    fig.savefig(folder / "mmr_distribution.png", dpi=160)
    plt.close(fig)

    dates = pd.to_datetime(matches["battle_time"], format="%Y%m%dT%H%M%S.%fZ", utc=True, errors="coerce").dropna()
    daily = dates.dt.floor("D").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    if not dates.empty:
        ax.bar(daily.index, daily.values)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(title=f"{len(matches):,} collected Ladder matches", xlabel="Battle date (UTC)", ylabel="Matches per day")
    fig.savefig(folder / "match_date_distribution.png", dpi=160)
    plt.close(fig)
    print(f"Saved MMR and match-date plots and the MMR distribution table in {folder}.", flush=True)


def merge_datasets(source_dir=None, output_dir=None):
    source_dir = Path(source_dir) if source_dir is not None else SOURCE_DIR
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    match_files = sorted(source_dir.glob("*/recent_matches.csv"))
    if not match_files:
        raise SystemExit(f"No MMR datasets found in {source_dir}.")
    matches = pd.concat([pd.read_csv(path) for path in match_files], ignore_index=True)
    matches = matches.reindex(columns=list(dict.fromkeys(MATCH_COLUMNS + list(matches.columns))))
    # Keep the more complete whole row: player orientation can differ by source.
    matches["end_state_fields"] = matches[END_STATE_COLUMNS].notna().sum(axis=1)
    matches = matches.sort_values("end_state_fields", ascending=False, kind="stable").drop_duplicates("battle_id")
    matches = matches.drop(columns="end_state_fields").sort_values(["battle_time", "battle_id"])
    player_files = [path.with_name("players.csv") for path in match_files if path.with_name("players.csv").exists()]
    players = (pd.concat([pd.read_csv(path) for path in player_files], ignore_index=True)
               if player_files else pd.DataFrame(columns=["tag", "mmr"]))
    # Prefer a known rating over an empty one; retain one approximate value per player.
    players["missing_rating"] = players["mmr"].isna()
    players = players.sort_values("missing_rating", kind="stable").drop_duplicates("tag")
    players = players[["tag", "mmr"]].sort_values("tag")
    output_dir.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_dir / "recent_matches.csv", index=False)
    players.to_csv(output_dir / "players.csv", index=False)
    save_plots(output_dir, rating_title="Observed Ladder ratings — all categories")

    save_card_usage(matches, output_dir)
    print(f"Merged {len(match_files)} categories: {len(matches):,} unique matches, {len(players):,} players.")
    print(f"Saved to {output_dir.resolve()}")


def main():
    merge_datasets()


if __name__ == "__main__":
    main()
