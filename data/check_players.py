"""Collect card/tower features and player ratings into CSVs. Edit settings below."""
import csv
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Lock
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjNkZTVlNGE4LTc5YWMtNGVmOC1hNjE4LWVhNWVjYjdhNzQ5NyIsImlhdCI6MTc5NDUyNDE2NCwic3ViIjoiZGV2ZWxvcGVyL2RmOWVhZDJjLTQxOTgtNGVhMC04NjI1LThmM2QyNDk5MmUyZSIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxNTguMTk1LjE3MC44NCIsIjE4OC4xNjcuMjUxLjU4Il0sInR5cGUiOiJjbGllbnQifV19.vl0-Al-GaIlLYqA9sUphUlgAFoiVV5ggPuzwPfz9SzQURLjAAafymiEJJG2Xr2RMSaAI3fRUYYUR9SpYKf9iYA"
MAX_PLAYERS = None  # None = every tag in the original CSV.
PLOTS_ONLY = False  # True = update graphs without calling the API.
REFRESH = False     # True = revisit players whose battle logs were already checked.
WORKERS = 8        # Maximum simultaneous player requests.
DELAY = 0.01       # Minimum gap between request starts across all workers.
SAVE_EVERY = 100
CSV_FILE = Path(__file__).with_name("ladder_30000.csv")
OUTPUT_DIR = Path(__file__).with_name("player_check")

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tools" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# API levels are relative to rarity. These offsets give the visible in-game level.
LEVEL_OFFSETS = {"common": 0, "rare": 2, "epic": 5, "legendary": 8, "champion": 10}
MATCH_COLUMNS = ["battle_id", "battle_time", "battle_type", "game_mode", "player_1_tag", "player_2_tag"]
for side in (1, 2):
    for card in range(1, 9):
        MATCH_COLUMNS += [f"player_{side}_card_{card}_{field}" for field in ("name", "level", "evolution", "hero")]
    MATCH_COLUMNS += [f"player_{side}_tower_name", f"player_{side}_tower_level"]
# Post-match information for analysis only; data.py does not use these as inputs.
END_STATE_COLUMNS = [f"player_{side}_{field}" for side in (1, 2) for field in (
    "crowns", "king_tower_hp", "princess_tower_1_hp", "princess_tower_2_hp", "elixir_leaked")]
MATCH_COLUMNS += END_STATE_COLUMNS + ["player_1_won"]
CARD_NAME_COLUMNS = [f"player_{side}_card_{index}_name" for side in (1, 2) for index in range(1, 9)]
REQUEST_LOCK = Lock()
STOP_REQUESTS = Event()
NEXT_REQUEST = 0.0


def get(path):
    global NEXT_REQUEST
    request = Request("https://api.clashroyale.com/v1" + path,
                      headers={"Authorization": "Bearer " + API_KEY.strip()})
    for attempt in range(3):
        while True:
            if STOP_REQUESTS.is_set():
                raise CancelledError()
            with REQUEST_LOCK:
                pause = NEXT_REQUEST - time.monotonic()
                if pause <= 0:
                    NEXT_REQUEST = time.monotonic() + DELAY
                    break
            STOP_REQUESTS.wait(pause)
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            cooldown = float(error.headers.get("Retry-After", 5))
            with REQUEST_LOCK:
                NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + cooldown)
            print(f"  HTTP {error.code}; pausing requests for {cooldown:g} seconds, retry {attempt + 1}/2...", flush=True)


def fetch_battlelogs(tags):
    # Keep only WORKERS requests in flight instead of queuing the entire CSV.
    global NEXT_REQUEST
    NEXT_REQUEST = 0.0
    STOP_REQUESTS.clear()
    executor = ThreadPoolExecutor(max_workers=WORKERS)
    remaining = iter(tags)
    futures = {}

    def submit_next():
        tag = next(remaining, None)
        if tag is not None:
            print(f"Fetching battle log for {tag}...", flush=True)
            futures[executor.submit(get, "/players/" + quote(tag, safe="") + "/battlelog")] = tag

    try:
        for _ in range(min(WORKERS, len(tags))):
            submit_next()
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                tag = futures.pop(future)
                yield tag, future
                submit_next()
    finally:
        STOP_REQUESTS.set()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)


def card_level(card):
    return card["level"] + LEVEL_OFFSETS[card["rarity"]]


def match_row(battle):
    if (battle.get("gameMode") or {}).get("name") != "Ladder":
        return None
    team, opponent = battle.get("team", []), battle.get("opponent", [])
    if len(team) != 1 or len(opponent) != 1:
        return None
    p1, p2 = team[0], opponent[0]
    if len(p1.get("cards", [])) != 8 or len(p2.get("cards", [])) != 8:
        return None
    change1, change2 = p1.get("trophyChange", 0) or 0, p2.get("trophyChange", 0) or 0
    if change1 > 0 >= change2:
        won = 1
    elif change2 > 0 >= change1:
        won = 0
    elif p1.get("crowns") is not None and p2.get("crowns") is not None and p1["crowns"] != p2["crowns"]:
        won = int(p1["crowns"] > p2["crowns"])
    else:
        return None  # Draw or no reliable winner; neither can be labeled 1/0.
    mode = battle.get("gameMode") or {}
    identity = [battle.get("battleTime"), battle.get("type"), mode.get("id"),
                battle.get("eventTag"), battle.get("tournamentTag"), sorted([[p1["tag"]], [p2["tag"]]])]
    row = {"battle_id": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
           "battle_time": battle.get("battleTime"), "battle_type": battle.get("type"),
           "game_mode": mode.get("name"), "player_1_tag": p1["tag"], "player_2_tag": p2["tag"]}
    for side, player in enumerate((p1, p2), 1):
        for index, card in enumerate(player["cards"], 1):
            prefix = f"player_{side}_card_{index}_"
            form = card.get("evolutionLevel", 0) or 0
            row.update({prefix + "name": card["name"], prefix + "level": card_level(card),
                        prefix + "evolution": int(bool(form & 1)), prefix + "hero": int(bool(form & 2))})
        tower = next(iter(player.get("supportCards") or []), None)
        row[f"player_{side}_tower_name"] = tower["name"] if tower else None
        row[f"player_{side}_tower_level"] = card_level(tower) if tower else None
        row[f"player_{side}_crowns"] = player.get("crowns")
        row[f"player_{side}_king_tower_hp"] = player.get("kingTowerHitPoints")
        row[f"player_{side}_elixir_leaked"] = player.get("elixirLeaked")
        # The API can omit destroyed towers. These are reported entries, not left/right.
        princess_hp = player.get("princessTowersHitPoints") or []
        for index in range(2):
            row[f"player_{side}_princess_tower_{index + 1}_hp"] = princess_hp[index] if index < len(princess_hp) else None
    row["player_1_won"] = won
    return row


def upgrade_match_columns(path):
    """Add new columns before appending, leaving old matches' new fields blank."""
    if not path.exists() or path.stat().st_size == 0:
        return MATCH_COLUMNS
    with path.open(newline="", encoding="utf-8") as file:
        previous = next(csv.reader(file))
    columns = MATCH_COLUMNS + [column for column in previous if column not in MATCH_COLUMNS]
    if previous == columns:
        return columns
    print(f"Updating CSV columns in {path}; older rows keep missing values blank...", flush=True)
    temporary = path.with_suffix(".schema.tmp")
    try:
        with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            for index, row in enumerate(csv.DictReader(source), 1):
                writer.writerow(row)
                if index % 100_000 == 0:
                    print(f"  Updated {index:,} saved matches...", flush=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return columns


def process_battles(tag, battles, players, seen, writer):
    if not isinstance(battles, list):
        raise ValueError("Expected a list of battles from the API.")
    reported = len(battles)
    battles = [battle for battle in battles if (battle.get("gameMode") or {}).get("name") == "Ladder"]
    for battle in battles:
        participants = battle.get("team", []) + battle.get("opponent", [])
        if not any(player.get("tag") == tag for player in participants):
            raise ValueError(f"A returned Ladder match does not contain the requested player {tag}.")
    players.setdefault(tag, None)
    skipped = reported - len(battles)
    for battle in battles:
        # Keep one Ladder rating per participant, never replace an existing value.
        for player in battle.get("team", []) + battle.get("opponent", []):
            player_tag = player["tag"]
            players.setdefault(player_tag, None)
            if (battle.get("gameMode") or {}).get("name") == "Ladder" and players[player_tag] is None:
                players[player_tag] = player.get("startingTrophies")
        row = match_row(battle)
        if row is None:
            skipped += 1
        elif row["battle_id"] not in seen:
            writer.writerow(row)
            seen.add(row["battle_id"])
    return skipped


def save_state(players, checked):
    print(f"Saving players.csv and progress ({len(checked):,} checked players)...", flush=True)
    temporary = OUTPUT_DIR / "players.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "mmr"])
        writer.writerows(sorted(players.items()))
    temporary.replace(OUTPUT_DIR / "players.csv")
    temporary = OUTPUT_DIR / "checked_players.tmp"
    temporary.write_text("\n".join(sorted(checked)), encoding="utf-8")
    temporary.replace(OUTPUT_DIR / "checked_players.txt")
    print("Saved player ratings and progress.", flush=True)


def filter_saved_matches(path):
    print("Checking saved matches for non-Ladder games...", flush=True)
    temporary = path.with_suffix(".tmp")
    removed = kept = 0
    with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for index, row in enumerate(reader, 1):
            if row["game_mode"] == "Ladder":
                writer.writerow(row)
                kept += 1
            else:
                removed += 1
            if index % 100_000 == 0:
                print(f"  Scanned {index:,} saved matches...", flush=True)
    if removed:
        temporary.replace(path)
        print(f"Removed {removed:,} saved non-Ladder matches.", flush=True)
    else:
        temporary.unlink()
    print(f"Saved matches checked: {kept:,} Ladder games kept.", flush=True)


def save_card_usage(matches, folder):
    matches = matches[["battle_id"] + CARD_NAME_COLUMNS].drop_duplicates("battle_id")
    usage = matches[CARD_NAME_COLUMNS].stack().value_counts().rename_axis("card").to_frame("decks")
    usage["usage_percent"] = usage["decks"] / (2 * len(matches)) * 100
    usage.to_csv(folder / "card_usage.csv")
    fig, ax = plt.subplots(figsize=(12, max(4, len(usage) * 0.24)), layout="constrained")
    ax.barh(usage.index, usage["usage_percent"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=9)
    ax.set(title=f"Card usage across {2 * len(matches):,} player decks",
           xlabel="Decks containing the card (%)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(folder / "card_usage.png", dpi=160)
    plt.close(fig)
    print(f"Saved card usage table and plot in {folder}.", flush=True)


def save_plots(original, players):
    print("Drawing MMR distribution...", flush=True)
    ratings = pd.Series(list(players.values()), dtype="float64").dropna()
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.hist(ratings, bins=30)
    ax.set(title=f"Approximate Ladder rating (startingTrophies)\n{len(ratings):,} players",
           xlabel="Rating (trophies)", ylabel="Players")
    fig.savefig(OUTPUT_DIR / "mmr_distribution.png", dpi=160)
    plt.close(fig)
    print("Saved mmr_distribution.png. Drawing match-date distribution...", flush=True)

    recent = pd.read_csv(OUTPUT_DIR / "recent_matches.csv", usecols=["battle_id", "battle_time"] + CARD_NAME_COLUMNS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for ax, frame, title in zip(axes, [original.drop_duplicates("battle_id"), recent],
                               ["Original CSV matches", "Collected Ladder matches (deduplicated)"]):
        dates = pd.to_datetime(frame["battle_time"], format="%Y%m%dT%H%M%S.%fZ", utc=True, errors="coerce").dropna()
        daily = dates.dt.floor("D").value_counts().sort_index()
        ax.set(title=f"{title}\n{len(dates):,} matches", xlabel="Battle date (UTC)", ylabel="Matches per day")
        if not dates.empty:
            ax.bar(daily.index, daily.values)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig(OUTPUT_DIR / "match_date_distribution.png", dpi=160)
    plt.close(fig)
    print("Saved match_date_distribution.png.", flush=True)
    save_card_usage(recent, OUTPUT_DIR)


def main():
    print(f"Reading player tags from {CSV_FILE}...", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(CSV_FILE)
    tags = pd.concat([original["player_tag"], original["opponent_tag"]]).dropna().str.strip().str.upper().drop_duplicates().tolist()[:MAX_PLAYERS]
    print(f"Found {len(tags):,} players. Loading saved ratings and progress...", flush=True)
    players, checked, seen = {}, set(), set()
    player_file, checked_file = OUTPUT_DIR / "players.csv", OUTPUT_DIR / "checked_players.txt"
    if player_file.exists():
        previous = pd.read_csv(player_file)
        for _, player in previous.iterrows():
            value = player.get("mmr")
            players[player["tag"]] = int(value) if pd.notna(value) else None
    if checked_file.exists():
        checked = set(checked_file.read_text(encoding="utf-8").splitlines())
    match_file = OUTPUT_DIR / "recent_matches.csv"
    append = match_file.exists() and "player_1_won" in pd.read_csv(match_file, nrows=0).columns
    if append:
        filter_saved_matches(match_file)
        match_columns = upgrade_match_columns(match_file)
        saved = pd.read_csv(match_file, usecols=["battle_id", "player_1_tag", "player_2_tag"])
        seen = set(saved["battle_id"])
        for tag in pd.concat([saved["player_1_tag"], saved["player_2_tag"]]).dropna().unique():
            players.setdefault(tag, None)
    cache_dir = OUTPUT_DIR / "cache"
    old_responses = list(cache_dir.glob("*.json"))
    if match_file.exists() and not append and not old_responses and PLOTS_ONLY:
        raise SystemExit("The old CSV has no card data and no saved responses remain. Run collection to fetch the card fields.")
    skipped = 0

    with match_file.open("a" if append else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=match_columns if append else MATCH_COLUMNS)
        if not append:
            writer.writeheader()
        # Convert old saved responses once, then remove them after saving the CSVs.
        if old_responses:
            print(f"Converting {len(old_responses):,} old saved responses into CSV rows...", flush=True)
        for path in old_responses:
            record = json.loads(path.read_text(encoding="utf-8"))
            skipped += process_battles(record["tag"], record["battles"], players, seen, writer)
            checked.add(record["tag"])
        file.flush()
        save_state(players, checked)
        for path in old_responses:
            path.unlink()
        if cache_dir.exists() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
        if old_responses:
            print(f"Converted and removed {len(old_responses):,} saved responses.", flush=True)

        pending = [] if PLOTS_ONLY else [tag for tag in tags if REFRESH or tag not in checked]
        if pending and not API_KEY.strip():
            raise SystemExit("Paste your API key into API_KEY at the top of this file.")
        print(f"{len(pending):,} players to fetch | {WORKERS} workers | {len(seen):,} saved matches", flush=True)
        started = time.monotonic()
        downloads = fetch_battlelogs(pending)
        try:
            for index, (tag, future) in enumerate(downloads, 1):
                try:
                    battles = future.result()
                except HTTPError as error:
                    if error.code in (401, 403):
                        raise SystemExit(f"HTTP {error.code}: check the API key and its allowed IP address.")
                    print(f"Skipping {tag}: HTTP {error.code}", flush=True)
                    continue
                except (URLError, TimeoutError, ConnectionError) as error:
                    print(f"Skipping {tag}: network error ({error})", flush=True)
                    continue
                before = len(seen)
                try:
                    player_skipped = process_battles(tag, battles, players, seen, writer)
                except ValueError as error:
                    print(f"Skipping {tag}: invalid battle log ({error})", flush=True)
                    continue
                skipped += player_skipped
                added = len(seen) - before
                duplicates = len(battles) - player_skipped - added
                file.flush()
                checked.add(tag)
                rating = players.get(tag)
                print(f"[{index:,}/{len(pending):,} | {index / len(pending):.1%}] {tag}: API returned {len(battles)} games | added {added} Ladder matches | duplicates {duplicates} | filtered/skipped {player_skipped} | MMR: {rating if rating is not None else 'unavailable'} | dataset total {len(seen):,}", flush=True)
                if index == 1 or index % SAVE_EVERY == 0 or index == len(pending):
                    save_state(players, checked)
                    elapsed = time.monotonic() - started
                    remaining = elapsed / index * (len(pending) - index)
                    print(f"Progress: {index:,}/{len(pending):,} players | elapsed {elapsed / 60:.1f} min | estimated remaining {remaining / 60:.1f} min", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. Saving completed CSV results; rerun to resume.", flush=True)
        finally:
            downloads.close()
            file.flush()
            save_state(players, checked)
    save_plots(original, players)
    print(f"Saved {len(players):,} player ratings and {len(seen):,} matches to {OUTPUT_DIR.resolve()}", flush=True)
    print(f"Filtered/skipped {skipped:,} entries: non-Ladder or without a resolved 8-card 1v1 result.", flush=True)


if __name__ == "__main__":
    main()
