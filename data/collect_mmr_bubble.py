"""Follow opponents from a starting player and collect one MMR-labelled dataset."""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from pathlib import Path
import random
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote

import check_players as api  # Reuse its API_KEY, request pacing and match format.
from merge_mmr_bubbles import save_plots

START_TAG = "20P2PGJ202"       # Example: "#YOURTAG"
START_MMR = 8000      # Set the approximate starting Ladder rating, e.g. 6000.
TARGET_MATCHES = 100_000
MAX_NEW_MATCHES_PER_PLAYER = 30
WORKERS = 8
DELAY = 0.01
EXTRA_START_TAGS = []  # Optional starting tags if every discovered path runs out.
OUTPUT_ROOT = Path(__file__).with_name("mmr_bubbles")


def normalize_tag(tag):
    return "#" + tag.strip().upper().lstrip("#")


def add_matches(tag, battles, players, seen, contributions, writer, target):
    if not isinstance(battles, list):
        raise ValueError("Expected a list of battles.")
    ladder = [battle for battle in battles if (battle.get("gameMode") or {}).get("name") == "Ladder"]
    for battle in ladder:
        if not any(player.get("tag") == tag for player in battle.get("team", []) + battle.get("opponent", [])):
            raise ValueError(f"A returned Ladder match does not contain {tag}.")
    added, opponents = 0, set()
    players.setdefault(tag, None)
    for battle in sorted(ladder, key=lambda item: item.get("battleTime", ""), reverse=True):
        if contributions.get(tag, 0) >= MAX_NEW_MATCHES_PER_PLAYER or len(seen) >= target:
            break
        row = api.match_row(battle)
        if row is None:
            continue
        if row["player_1_tag"] != tag:
            row = api.match_row({**battle, "team": battle["opponent"], "opponent": battle["team"]})
        for player in battle["team"] + battle["opponent"]:
            player_tag = player["tag"]
            if players.get(player_tag) is None:
                players[player_tag] = player.get("startingTrophies")
            if player_tag != tag:
                opponents.add(player_tag)
        if row["battle_id"] in seen:
            continue  # Duplicates do not consume this player's 30-new-match quota.
        writer.writerow(row)
        seen.add(row["battle_id"])
        contributions[tag] = contributions.get(tag, 0) + 1
        added += 1
    return added, opponents


def save_progress(folder, players, checked, contributions, heads):
    temporary = folder / "players.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "mmr"])
        writer.writerows(sorted(players.items()))
    temporary.replace(folder / "players.csv")
    temporary = folder / "progress.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "added", "checked"])
        writer.writerows((tag, contributions.get(tag, 0), int(tag in checked)) for tag in sorted(set(contributions) | checked))
    temporary.replace(folder / "progress.csv")
    temporary = folder / "walk_heads.tmp"
    temporary.write_text("\n".join(dict.fromkeys(heads)), encoding="utf-8")
    temporary.replace(folder / "walk_heads.txt")


def collect_bubble(start_tag, start_mmr, folder, target=TARGET_MATCHES):
    start_tag = normalize_tag(start_tag)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    players, checked, contributions, seen = {}, set(), {}, set()
    if (folder / "players.csv").exists():
        with (folder / "players.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                players[row["tag"]] = int(float(row["mmr"])) if row["mmr"] else None
    if (folder / "progress.csv").exists():
        with (folder / "progress.csv").open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                contributions[row["tag"]] = int(row["added"])
                if row["checked"] == "1":
                    checked.add(row["tag"])
    match_file = folder / "recent_matches.csv"
    match_columns = api.upgrade_match_columns(match_file)
    if match_file.exists():
        contributions = {}  # Rebuild quotas from durable rows after an interrupted save.
        with match_file.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["battle_id"] in seen:
                    continue
                seen.add(row["battle_id"])
                owner = row["player_1_tag"]  # Stored as the player whose log supplied the row.
                contributions[owner] = contributions.get(owner, 0) + 1
                players.setdefault(row["player_1_tag"], None)
                players.setdefault(row["player_2_tag"], None)
    checked.update(tag for tag, count in contributions.items() if count >= MAX_NEW_MATCHES_PER_PLAYER)
    heads = deque((folder / "walk_heads.txt").read_text(encoding="utf-8").splitlines() if (folder / "walk_heads.txt").exists() else [])
    if start_tag not in checked and start_tag not in heads:
        heads.appendleft(start_tag)
    if players.get(start_tag) is None:
        players[start_tag] = start_mmr
    if len(seen) >= target:
        print(f"Already have {len(seen):,}/{target:,} matches in {folder}. No requests needed.", flush=True)
        return len(seen)
    if not api.API_KEY.strip():
        raise SystemExit("Set API_KEY in data/check_players.py first.")

    api.DELAY = DELAY
    api.NEXT_REQUEST = 0.0
    api.STOP_REQUESTS.clear()
    executor = ThreadPoolExecutor(max_workers=WORKERS)
    futures, attempted = {}, set()
    extras = deque(normalize_tag(tag) for tag in EXTRA_START_TAGS)
    rng = random.Random(47)
    initial_matches, started = len(seen), time.monotonic()

    def available(tag):
        return tag not in checked and tag not in attempted and contributions.get(tag, 0) < MAX_NEW_MATCHES_PER_PLAYER

    def next_tag():
        while heads:
            candidate = heads.popleft()
            if available(candidate):
                return candidate
        candidates = [tag for tag in players if available(tag)]
        if candidates:
            candidate = rng.choice(candidates)
            print(f"Using discovered fallback {candidate}.", flush=True)
            return candidate
        if not futures:
            while extras:
                candidate = extras.popleft()
                if available(candidate):
                    players.setdefault(candidate, None)
                    print(f"Paths exhausted; using extra starting tag {candidate}.", flush=True)
                    return candidate
        return None

    def checkpoint(file):
        file.flush()
        remaining_heads = [tag for tag in list(heads) + list(futures.values())
                           if tag not in checked and contributions.get(tag, 0) < MAX_NEW_MATCHES_PER_PLAYER]
        # Windows readers can briefly prevent a progress file from being replaced.
        for attempt in range(10):
            try:
                save_progress(folder, players, checked, contributions, remaining_heads[:WORKERS])
                return True
            except PermissionError as error:
                if attempt == 9:
                    print(f"Progress save delayed by Windows ({error}). Match CSV is flushed; progress will be retried at the next checkpoint.", flush=True)
                    return False
                time.sleep(0.1)

    print(f"Starting category {start_mmr} from {start_tag}: {len(seen):,}/{target:,} matches, up to {WORKERS} paths.", flush=True)
    with match_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=match_columns)
        if file.tell() == 0:
            writer.writeheader()
        try:
            while len(seen) < target:
                while len(futures) < WORKERS:
                    tag = next_tag()
                    if tag is None:
                        break
                    attempted.add(tag)
                    contributions.setdefault(tag, 0)
                    future = executor.submit(api.get, "/players/" + quote(tag, safe="") + "/battlelog")
                    futures[future] = tag
                    print(f"Fetching {tag} | active paths {len(futures)}/{WORKERS} | {len(seen):,}/{target:,} matches", flush=True)
                if not futures:
                    print("All discovered paths are exhausted. Add EXTRA_START_TAGS or choose a new starting player.", flush=True)
                    break
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    if len(seen) >= target:
                        break
                    tag = futures.pop(future)
                    try:
                        battles = future.result()
                        added, opponents = add_matches(tag, battles, players, seen, contributions, writer, target)
                    except HTTPError as error:
                        if error.code in (401, 403):
                            raise SystemExit(f"HTTP {error.code}: check API_KEY and its allowed IP address.")
                        if error.code == 404:
                            checked.add(tag)
                        print(f"Dead end {tag}: HTTP {error.code}.", flush=True)
                        checkpoint(file)
                        continue
                    except (URLError, TimeoutError, ConnectionError, ValueError) as error:
                        print(f"Dead end {tag}: {error}", flush=True)
                        checkpoint(file)
                        continue
                    checked.add(tag)
                    candidates = [opponent for opponent in sorted(opponents) if available(opponent) and opponent not in heads]
                    if candidates:
                        heads.append(rng.choice(candidates))  # One continuation, not all opponents.
                    else:
                        print(f"No unvisited opponent from {tag}; this path will use a fallback.", flush=True)
                    checkpoint(file)
                    gained = len(seen) - initial_matches
                    elapsed = time.monotonic() - started
                    eta = f"{(target - len(seen)) * elapsed / gained / 60:.1f} min" if gained else "unknown"
                    print(f"{tag}: API returned {len(battles)} games | added {added} | player contribution {contributions[tag]}/{MAX_NEW_MATCHES_PER_PLAYER} | dataset {len(seen):,}/{target:,} ({len(seen) / target:.1%}) | ETA {eta}", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. Saving CSVs and the current paths; rerun to resume.", flush=True)
        finally:
            api.STOP_REQUESTS.set()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            progress_saved = checkpoint(file)
    if progress_saved:
        save_plots(folder)
    else:
        print("Some progress files are still pending. Rerun to resume from the saved match CSV.", flush=True)
    print(f"Saved {len(seen):,}/{target:,} unique matches in {folder.resolve()} | {len(players):,} players observed.", flush=True)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=START_TAG, help="Starting player tag (defaults to START_TAG).")
    parser.add_argument("--mmr", type=int, default=START_MMR, help="Starting MMR category (defaults to START_MMR).")
    parser.add_argument("--require-target", action="store_true", help="Exit with status 3 if collection ends below the match target.")
    args = parser.parse_args()
    if not args.tag.strip() or args.mmr is None:
        parser.error("Pass --tag and --mmr, or set START_TAG and START_MMR in the script.")
    saved = collect_bubble(args.tag, args.mmr, OUTPUT_ROOT / str(args.mmr))
    return 3 if args.require_target and saved < TARGET_MATCHES else 0


if __name__ == "__main__":
    raise SystemExit(main())
