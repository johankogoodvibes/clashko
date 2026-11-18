# Train Clash Royale predictions on the NVIDIA GPU

## Predict a custom deck matchup

Double-click `run-predictor.cmd`, or run it from PowerShell:

```powershell
.\run-predictor.cmd
```

The desktop app uses the existing project environment; no extra packages are
needed. Select eight unique cards for **Your deck** and **Opponent's deck**.
Start typing a card name to automatically show matching suggestions. Keep typing
to narrow the list, click a suggestion, or use Up/Down and Enter to select one.
Escape dismisses suggestions. Set each card's level (1–16),
Evolution and Hero flags, and each deck's tower troop and level. Enable upgrade
flags only for the card versions you actually use. Click **Calculate probability**
to display both players' estimated win probabilities. **Load example** fills a
sample matchup; **Swap decks** reverses the sides. You can save and load deck
matchups as JSON files.

The app loads `outputs/model.h5` by default. **Browse...** selects another saved
H5 model. It loads the saved architecture and weights directly, so editing
`model.py` does not change predictions until you train and save a new model.
The first calculation loads TensorFlow and may take a few seconds; later
calculations reuse the model. Inference supports CPU or the configured GPU.

Your deck is always player 1. The displayed win chance is softmax output index
1, matching the labels in `train.py`; index 0 is the opponent's win chance.
These are model estimates. Swapping the decks runs a new prediction, and the
model may give different estimates depending on which deck is player 1.
Inputs use exactly the fixed card/tower order and feature normalization from
`data.py`. Duplicate cards, unknown names and invalid levels are rejected.

The interface is in `deck_predictor.py`; reusable encoding and model inference
are in `predict.py`. Run directly with `.\.venv\python.exe .\deck_predictor.py`.

This project uses native Windows, Python 3.10, TensorFlow 2.10.1, Keras 2.10,
CUDA 11.2, and cuDNN 8.1. TensorFlow 2.10 is the last release with native
Windows CUDA support: https://www.tensorflow.org/install/pip#windows-native

## Run training

Open PowerShell in this folder and run:

```powershell
.\run-training.cmd
```

The launcher uses `.venv\python.exe` and the environment's CUDA libraries.
You can also run `.\.venv\python.exe .\train.py` directly. The script configures
CUDA before importing Keras and prints the detected GPU and model weights device.
In VS Code, select **Run and Debug > Train on GPU** and press **F5**.

The default run loads all rows from your collected dataset at
`data/player_check/recent_matches.csv` and trains for 10 epochs. Training
reads the local CSV; it does not download matches or call the API.

Training settings are at the top of `train.py`. The CSV path and row limit
are in `data.py`.
`MAX_ROWS = 50_000` caps the dataset after filtering, including the
`MIN_TROPHIES` filter in `data.py`. Set `MAX_ROWS = None` to keep every eligible match.
The default split is 80% training, 10% validation, and 10% reserved for testing.
`train_experiments.py` runs the same training pipeline for each model configuration
in `RUNS`. Each run uses the same dataset and splits.
The training scripts use the validation split for metrics and reserve the test split
for later evaluation.

Each run saves these files in `OUTPUT_DIR` (default `outputs/`):

- `history.png`: training and validation loss and accuracy plots.
- `history.csv`: the metrics for every epoch.
- `model.h5`: the trained Keras model.

Files in the chosen output folder are overwritten on subsequent runs.
To redraw a saved plot without training:

```powershell
.\.venv\python.exe .\plot_history.py
# Or pass a different run's history:
.\.venv\python.exe .\plot_history.py .\outputs\experiment-2\history.csv
```

## Edit the model and data

- `model.py`: three stages. Dense(16, ReLU) processes the four features of each
  card with weights shared across cards. Flatten and Dense(100, ReLU) then learn
  interactions within a deck, keeping each card's identity through its fixed
  position. TimeDistributed shares this whole deck encoder between both players.
  Their two 100-value representations are joined, followed by Dense(100, ReLU)
  for the matchup and Dense(2, softmax) for the winner. The model is compiled
  with categorical cross-entropy and Adam(0.001).
- `data.py`: the card/tower vocabularies, local dataset loading, and input encoding.
- `train.py`: the 80/10/10 split, one-hot labels, training, and saved results.
- `plot_history.py`: the history plots.
- `gpu_runtime.py`: native Windows CUDA library setup and GPU detection.

Inputs have shape `(matches, 2, 127, 4)`: the fixed 123 card rows followed by
Tower Princess, Cannoneer, Dagger Duchess, and Royal Chef. Each row holds
`[presence, evolution, hero, in-game level / 16]`, filled from the collected CSV.
Tower rows use presence and normalized level, with evolution and hero set to
zero. Unreported towers stay zero; a reported tower with an unknown level uses
zero for that level. The shared model accepts the larger input automatically.
`y` is 1 for a player-1 win and 0 for a player-2 win. The loader retains Ladder
matches, removes duplicate battle IDs, and skips incomplete card records.

`MAX_ELIXIR_LEAKED` in `data.py` defaults to `10.0`. Training retains matches
where both players leaked strictly less than this amount. Matches with a
missing or invalid leaked-elixir value are excluded, and the loader prints
how many matches it kept and removed. Set it to `None` to disable the filter.
The filter is automatically skipped for CSVs without both leaked-elixir columns.

The original notebook and its saved outputs are preserved in
`archive/clash_royale_ladder_template.ipynb`.

## Collect the new match dataset

`data/check_players.py` uses tags from both sides of `data/ladder_30000.csv`
and fetches their recent battle logs. Put your API key into `API_KEY` near the
start of that file, then run:

```powershell
.\.venv\python.exe .\data\check_players.py
```

Settings are hardcoded at the top: `MAX_PLAYERS`, `PLOTS_ONLY`, `REFRESH`,
`WORKERS`, `DELAY`, and `SAVE_EVERY`. The collector requests battle logs only and stores
structured CSV rows. Old saved JSON responses are converted into these CSVs
and removed automatically; new responses are processed in memory.
The terminal shows each player request, received/added/skipped match counts,
retries, saves, plot generation, and periodic elapsed/remaining time estimates.
`WORKERS = 8` allows eight concurrent requests. Results are processed as they
arrive, while CSV writes and match deduplication stay in the main thread.
`DELAY` spaces request starts across all workers; rate-limit responses pause
new requests together. Completed players remain resumable after interruption.
All returned Ladder matches are considered, with no per-player match cap.
Returned Ladder matches must include the requested player. The terminal
distinguishes the API response size, newly added matches, and the total dataset
size. Matches encountered through multiple players' logs are deduplicated.

Files in `data/player_check/`:

- `players.csv`: exactly `tag,mmr`. The approximate rating is the first
  `startingTrophies` reported for that participant in a `Ladder` battle. A
  recorded value is kept on later runs. Missing values stay blank and can be
  filled if another battle later supplies a rating. Ranked ratings and
  tournament/challenge win counts are not mixed into this column.
- `recent_matches.csv`: one row per resolved Ladder 1v1 match. Each player's eight
  cards have name, in-game level, evolution flag, and hero flag; their tower
  troop has name and in-game level. `player_1_won` is 1 for a player-1 win and
  0 for a player-2 win. These are the 69 requested data fields, plus six
  metadata columns: battle ID, timestamp, type, mode, and both player tags.
  Ten additional post-match columns store final tower state and leaked elixir.
- `checked_players.txt`: tags whose battle logs were fetched, for resuming.
- `mmr_distribution.png` and `match_date_distribution.png`: distributions
  from the saved CSVs, including original versus collected match dates.
- `card_usage.png` and `card_usage.csv`: card usage ranked by the percentage
  of player decks containing each card, counting both sides of each unique match.

Card and tower levels use visible in-game levels (the rarity offset is added
where required). Evolution and Hero are decoded from the battle card's
`evolutionLevel` bits; card capability/ownership fields are not used. Tower
fields stay blank if the API does not report a tower troop.

Both collectors also save, for each side (`player_1_` / `player_2_`):

- `crowns`: the player's final crown score.
- `king_tower_hp`: remaining King Tower hit points at the end.
- `princess_tower_1_hp` and `princess_tower_2_hp`: reported remaining Princess
  Tower hit points, in API order. These are not left/right positions. Omitted
  entries stay blank; no zero HP is invented for an omitted tower.
- `elixir_leaked`: the API's `elixirLeaked` value, keeping its decimal precision.

These fields are saved for analysis and are not used as inputs by `data.py`.
Missing values stay blank; reported zeros are preserved. On the next collector
run, existing CSVs gain these columns with blanks for older matches. Existing
matches are still deduplicated, so this does not refetch or backfill their state.
The merger accepts old and new CSVs and prefers the row with more reported
post-match fields when a battle appears in multiple categories.

Only battles with `gameMode.name == "Ladder"` are stored; Ranked / Path of
Legends matches are excluded. The next run also removes any previously saved
non-Ladder rows before appending new matches.

Matches seen in both participants' logs are saved once. 2v2s, multi-deck duels,
boat defenses, and matches without a reliable 1/0 result are skipped because
these cannot fit the two-player, eight-card schema. No date or rating-range
filter is applied. `REFRESH = True` revisits checked tags while retaining
existing ratings and deduplicating matches. `PLOTS_ONLY = True` redraws graphs
from CSVs without API calls (and also converts any remaining old responses).

The training loader in `data.py` reads this collected match CSV by default.

## Collect an MMR category by following opponents

For a separate collection method, edit the settings at the top of
`data/collect_mmr_bubble.py`:

```python
START_TAG = "#YOURTAG"
START_MMR = 6000
TARGET_MATCHES = 10_000
MAX_NEW_MATCHES_PER_PLAYER = 30
WORKERS = 8
```

It reuses `API_KEY` from `data/check_players.py`. Pass the starting tag and MMR
as command-line arguments:

```powershell
.\.venv\python.exe .\data\collect_mmr_bubble.py --tag "#YOURTAG" --mmr 6000
```

Omit these arguments to use `START_TAG` and `START_MMR` from the script.

The first request discovers opponents of the starting player. Then up to eight
paths run concurrently. Each completed player chooses one unvisited opponent
as its continuation; the other discovered tags are fallbacks at dead ends,
rather than all being scheduled at once. `EXTRA_START_TAGS` can provide more
starting points if every known path runs out. The collector stops at the target
number of unique matches, or reports exhaustion if it cannot reach it.

Each fetched player contributes at most 30 NEW Ladder matches. Duplicate games
do not consume that quota. A player can also appear as an opponent in rows
collected from other players. Player 1 is always the player whose log supplied
the row, so the contribution quota can be reconstructed from CSVs on resume.
The winner label and card/tower features follow that orientation.

Keep the MMR folder when restarting. The collector reloads saved battle IDs,
checked players and contribution counts, then appends previously unseen games.
The target is the total number of matches in that folder across all runs.
Once it reaches `TARGET_MATCHES`, increase that setting to collect more.
Checked players remain skipped while the walk continues through other players.

For the example above, files are saved under `data/mmr_bubbles/6000/`:

- `recent_matches.csv` and `players.csv`: the same match and rating schemas as
  the original collector.
- `progress.csv` and `walk_heads.txt`: checked players, contribution counts,
  and current paths for resuming after Ctrl+C.
- `mmr_distribution.png` and `match_date_distribution.png` show observed ratings
  and dates in this category. `mmr_distribution.csv` contains player counts by
  exact MMR. The date graph uses timestamps from `recent_matches.csv`.

Responses are processed in memory. The folder name labels the starting MMR;
the walk can drift, and actual observed ratings are kept in `players.csv`.

Run the merge manually when you want to update `data/combined_dataset/` and
its combined MMR and match-date distribution plots:

```powershell
.\.venv\python.exe .\data\merge_mmr_bubbles.py
```

This writes `data/combined_dataset/recent_matches.csv` and `players.csv`,
deduplicating battles shared between categories and keeping one known rating
per player. It also writes `mmr_distribution.csv`, `mmr_distribution.png`
and `match_date_distribution.png`. `card_usage.csv` and `card_usage.png` rank every observed
card by the number and percentage of player decks containing it, counting both
sides of each unique match. To train on that collection, set `CSV_FILE` in
`train.py` to the combined match CSV.

## Install or restore the environment

Install or restore the project environment from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-gpu.ps1
```

The installer downloads a portable environment manager into `.tools`, creates
`.venv`, installs the Python dependencies, and verifies GPU computation and
one model training batch. The plain Python workflow does not require Jupyter.

Recheck GPU operation from PowerShell:

```powershell
.\.venv\python.exe .\check_gpu.py
```

The model is small; GPU use can be brief and may not be faster
than CPU training for this model.
