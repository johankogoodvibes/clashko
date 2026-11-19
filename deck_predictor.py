"""Desktop deck builder. Run with py -3.10 deck_predictor.py."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from data import MAX_LEVEL, card_names, tower_names
from predict import DEFAULT_MODEL, MatchupPredictor, encode_matchup


def make_deck(names, evolution=(), heroes=()):
    return {
        "cards": [{"name": name, "level": 16, "evolution": name in evolution, "hero": name in heroes}
                  for name in names],
        "tower": {"name": "Tower Princess", "level": 16},
    }


PRESET_DECKS = {
    # https://royaleapi.com/decks/stats/cannon-ev1,fireball,hog-rider,ice-golem,ice-spirit,musketeer-hero,skeletons-ev1,the-log
    "Hog Cycle 2.6": make_deck([
        "Hog Rider", "Musketeer", "Ice Golem", "Skeletons",
        "Ice Spirit", "Cannon", "Fireball", "The Log",
    ], evolution={"Cannon", "Skeletons"}, heroes={"Musketeer"}),
    # https://royaleapi.com/decks/stats/baby-dragon-ev1,battle-healer,cannon,electro-dragon-ev1,elixir-golem,knight-hero,mother-witch,tornado
    "Elixir Golem": make_deck([
        "Elixir Golem", "Battle Healer", "Electro Dragon", "Baby Dragon",
        "Knight", "Tornado", "Mother Witch", "Cannon",
    ], evolution={"Electro Dragon", "Baby Dragon"}, heroes={"Knight"}),
    # https://royaleapi.com/decks/stats/baby-dragon-ev1,barbarian-barrel,golem,lightning,lumberjack-ev1,mega-minion-hero,night-witch,tornado
    "Golem": make_deck([
        "Golem", "Night Witch", "Baby Dragon", "Mega Minion",
        "Lumberjack", "Lightning", "Tornado", "Barbarian Barrel",
    ], evolution={"Lumberjack", "Baby Dragon"}, heroes={"Mega Minion"}),
    # https://royaleapi.com/decks/stats/goblin-barrel-ev1,goblin-gang,ice-spirit,inferno-tower,knight-hero,princess-ev1,rocket,the-log
    "Classic Logbait": make_deck([
        "Goblin Barrel", "Princess", "Knight", "Goblin Gang",
        "Ice Spirit", "Inferno Tower", "Rocket", "The Log",
    ], evolution={"Goblin Barrel", "Princess"}, heroes={"Knight"}),
    # https://royaleapi.com/decks/stats/arrows,elixir-golem,goblin-curse,lightning,night-witch,skeleton-army-ev1,skeleton-king,witch-ev1
    "Elixir Golem - Skeleton King": make_deck([
        "Elixir Golem", "Skeleton King", "Night Witch", "Witch",
        "Skeleton Army", "Goblin Curse", "Lightning", "Arrows",
    ], evolution={"Witch", "Skeleton Army"}),
    # https://royaleapi.com/decks/stats/goblin-curse,golem,inferno-dragon-ev1,miner,night-witch,skeleton-army-ev1,skeleton-king,zap
    "Golem - Skeleton King Miner": make_deck([
        "Golem", "Skeleton King", "Miner", "Night Witch",
        "Inferno Dragon", "Skeleton Army", "Goblin Curse", "Zap",
    ], evolution={"Inferno Dragon", "Skeleton Army"}),
    "My deck - Golden Knight": make_deck([
        "Golden Knight", "Goblin Barrel", "Dart Goblin", "Witch",
        "Cannon", "Rocket", "Arrows", "Skeleton Army",
    ], evolution={"Dart Goblin", "Witch"}),
    "Risko's deck": make_deck([
        "Hog Rider", "Arrows", "Freeze", "Mini P.E.K.K.A",
        "Skeleton Army", "Wizard", "Rage", "Inferno Dragon",
    ], evolution={"Inferno Dragon"}, heroes={"Mini P.E.K.K.A", "Wizard"}),
}


def example_matchup():
    return {
        "player": make_deck(["Hog Rider", "Earthquake", "Firecracker", "Knight", "Skeletons",
                             "Ice Spirit", "The Log", "Tesla"], {"Firecracker", "Knight"}),
        "opponent": make_deck(["Giant", "Prince", "Dark Prince", "Mega Minion", "Electro Wizard",
                               "Fireball", "Zap", "Miner"]),
    }


class CardCombobox(ttk.Combobox):
    """Show live suggestions without moving keyboard focus out of the entry."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, values=card_names, **kwargs)
        self._popup = None
        self._suggestions = None
        self.bind("<KeyRelease>", self._typing)
        self.bind("<Down>", lambda event: self._move(1))
        self.bind("<Up>", lambda event: self._move(-1))
        self.bind("<Return>", self._accept)
        self.bind("<Escape>", self._dismiss)
        self.bind("<Tab>", self._hide)
        self.bind("<ButtonPress-1>", self._hide)
        self.bind("<<ComboboxSelected>>", self._hide)
        self.bind("<FocusOut>", lambda event: self.after_idle(self._check_focus))
        self.bind("<Unmap>", self._hide)

    def _typing(self, event):
        if event.keysym in ("Up", "Down", "Return", "Escape", "Tab"):
            return
        if event.char or event.keysym in ("BackSpace", "Delete"):
            self._show()

    def _show(self):
        query = self.get().strip().casefold()
        matches = [name for name in card_names if query in name.casefold()]
        self.configure(values=matches)
        if not matches:
            self._hide()
            return
        if self._popup is None:
            self._popup = tk.Toplevel(self)
            self._popup.withdraw()
            self._popup.overrideredirect(True)
            self._popup.transient(self.winfo_toplevel())
            self._suggestions = tk.Listbox(
                self._popup, exportselection=False, takefocus=False,
                font=("Segoe UI", 10), relief="solid", borderwidth=1,
                selectmode="browse", selectbackground="#245a81", selectforeground="white",
            )
            scrollbar = ttk.Scrollbar(self._popup, command=self._suggestions.yview)
            self._suggestions.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            self._suggestions.pack(side="left", fill="both", expand=True)
            self._suggestions.bind("<ButtonPress-1>", self._click_suggestion)
        self._suggestions.delete(0, tk.END)
        self._suggestions.insert(tk.END, *matches)
        self._suggestions.configure(height=min(6, len(matches)))
        self._suggestions.selection_set(0)
        self._suggestions.activate(0)
        self._popup.update_idletasks()
        height = self._suggestions.winfo_reqheight()
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()
        if y + height > self.winfo_screenheight():
            y = self.winfo_rooty() - height
        self._popup.geometry(f"{self.winfo_width()}x{height}+{x}+{max(0, y)}")
        self._popup.deiconify()
        self._popup.lift()
        self.focus_set()

    def _move(self, step):
        if self._popup is None or not self._popup.winfo_viewable():
            self._show()
        else:
            current = self._suggestions.curselection()
            index = ((current[0] if current else 0) + step) % self._suggestions.size()
            self._suggestions.selection_clear(0, tk.END)
            self._suggestions.selection_set(index)
            self._suggestions.activate(index)
            self._suggestions.see(index)
        return "break"

    def _accept(self, event=None):
        if self._popup is not None and self._popup.winfo_viewable():
            selected = self._suggestions.curselection()
            if selected:
                self.set(self._suggestions.get(selected[0]))
                self.icursor(tk.END)
            self._hide()
            self.event_generate("<<ComboboxSelected>>")
            return "break"

    def _click_suggestion(self, event):
        index = self._suggestions.nearest(event.y)
        self._suggestions.selection_clear(0, tk.END)
        self._suggestions.selection_set(index)
        self._accept()
        self.focus_set()
        return "break"

    def _hide(self, event=None):
        if self._popup is not None:
            self._popup.withdraw()

    def _dismiss(self, event):
        self._hide()
        return "break"

    def _check_focus(self):
        # Native ttk popdowns have Tcl paths but no Python widget wrappers.
        if self.winfo_exists() and str(self.tk.call("focus")) != str(self):
            self._hide()


class DeckEditor(ttk.LabelFrame):
    def __init__(self, parent, title, changed):
        super().__init__(parent, text=title, padding=12)
        self.columnconfigure(1, weight=1)
        self.cards = []
        presets = ttk.Frame(self)
        presets.grid(row=0, column=0, columnspan=5, sticky="ew", pady=(0, 8))
        presets.columnconfigure(1, weight=1)
        ttk.Label(presets, text="Preset").grid(row=0, column=0, padx=(0, 8))
        preset = tk.StringVar(value="Choose a deck...")
        preset_picker = ttk.Combobox(
            presets, textvariable=preset, values=list(PRESET_DECKS), state="readonly",
        )
        preset_picker.grid(row=0, column=1, sticky="ew")
        preset_picker.bind("<<ComboboxSelected>>", lambda event: self.set(PRESET_DECKS[preset.get()]))
        for column, heading in enumerate(("", "Card (type to search)", "Level", "Evo", "Hero")):
            ttk.Label(self, text=heading).grid(row=1, column=column, padx=5, pady=(0, 8), sticky="w")
        for row in range(1, 9):
            name = tk.StringVar()
            level = tk.StringVar(value="14")
            evolution = tk.BooleanVar(value=False)
            hero = tk.BooleanVar(value=False)
            ttk.Label(self, text=str(row)).grid(row=row + 1, column=0, padx=(0, 6))
            combo = CardCombobox(self, textvariable=name, width=23)
            combo.grid(row=row + 1, column=1, padx=5, pady=3, sticky="ew")
            ttk.Spinbox(self, from_=1, to=MAX_LEVEL, width=4, textvariable=level).grid(
                row=row + 1, column=2, padx=5)
            ttk.Checkbutton(self, variable=evolution).grid(row=row + 1, column=3, padx=6)
            ttk.Checkbutton(self, variable=hero).grid(row=row + 1, column=4, padx=6)
            self.cards.append((name, level, evolution, hero))
            for variable in self.cards[-1]:
                variable.trace_add("write", changed)
        ttk.Separator(self).grid(row=10, column=0, columnspan=5, sticky="ew", pady=8)
        ttk.Label(self, text="Tower troop").grid(row=11, column=0, columnspan=2, sticky="w")
        self.tower = tk.StringVar(value=tower_names[0])
        self.tower_level = tk.StringVar(value="14")
        ttk.Combobox(self, textvariable=self.tower, values=tower_names, state="readonly", width=23).grid(
            row=12, column=0, columnspan=2, padx=(0, 5), pady=6, sticky="ew")
        ttk.Spinbox(self, from_=1, to=MAX_LEVEL, width=4, textvariable=self.tower_level).grid(
            row=12, column=2, padx=5)
        self.tower.trace_add("write", changed)
        self.tower_level.trace_add("write", changed)

    def get(self):
        lookup = {name.casefold(): name for name in card_names}
        return {
            "cards": [{"name": lookup.get(name.get().strip().casefold(), name.get().strip()),
                       "level": level.get(), "evolution": evo.get(), "hero": hero.get()}
                      for name, level, evo, hero in self.cards],
            "tower": {"name": self.tower.get(), "level": self.tower_level.get()},
        }

    def set(self, deck):
        for variables, card in zip(self.cards, deck["cards"]):
            for variable, field in zip(variables, ("name", "level", "evolution", "hero")):
                variable.set(card.get(field, False))
        self.tower.set(deck["tower"]["name"])
        self.tower_level.set(deck["tower"]["level"])
        for child in self.winfo_children():
            if isinstance(child, ttk.Combobox) and str(child.cget("state")) != "readonly":
                child.configure(values=card_names)


class DeckPredictorApp:
    def __init__(self, root):
        self.root = root
        root.title("Clash Royale - Deck Matchup Predictor")
        root.geometry("1080x750")
        root.minsize(990, 730)
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 6))
        style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Result.TLabel", font=("Segoe UI", 18, "bold"))
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.predictor = MatchupPredictor()
        self.future = None
        self.revision = 0

        body = ttk.Frame(root, padding=16)
        body.pack(fill="both", expand=True)
        body.columnconfigure((0, 1), weight=1)
        ttk.Label(body, text="Deck matchup predictor", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(body, text="Choose eight cards per deck, their levels and upgrades, and a tower troop.").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))

        model_bar = ttk.Frame(body)
        model_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        model_bar.columnconfigure(1, weight=1)
        ttk.Label(model_bar, text="Trained model").grid(row=0, column=0, padx=(0, 10))
        self.model_path = tk.StringVar(value=str(DEFAULT_MODEL))
        self.model_path.trace_add("write", self._changed)
        ttk.Entry(model_bar, textvariable=self.model_path).grid(row=0, column=1, sticky="ew")
        ttk.Button(model_bar, text="Browse...", command=self._browse_model).grid(row=0, column=2, padx=(10, 0))

        self.player = DeckEditor(body, "Your deck", self._changed)
        self.player.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
        self.opponent = DeckEditor(body, "Opponent's deck", self._changed)
        self.opponent.grid(row=3, column=1, sticky="nsew", padx=(8, 0))

        actions = ttk.Frame(body)
        actions.grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)
        for text, command in (("Load example", self._example), ("Swap decks", self._swap),
                              ("Save matchup...", self._save), ("Load matchup...", self._load)):
            ttk.Button(actions, text=text, command=command).pack(side="left", padx=(0, 8))
        self.predict_button = ttk.Button(actions, text="Calculate probability", command=self._predict)
        self.predict_button.pack(side="right")

        self.result = tk.StringVar(value="Your win chance: --     Opponent: --")
        ttk.Label(body, textvariable=self.result, style="Result.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w")
        self.bar = ttk.Progressbar(body, maximum=100)
        self.bar.grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
        self.status = tk.StringVar(value="Build both decks or load the example, then calculate.")
        ttk.Label(body, textvariable=self.status, wraplength=1000).grid(
            row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(body, text="Your deck is player 1. Probabilities are the saved model's estimates. "
                  "Evo = Evolution; enable Evo/Hero only for the version of a card you use.",
                  wraplength=1000).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))
        root.protocol("WM_DELETE_WINDOW", self._close)

    def _changed(self, *_):
        self.revision += 1
        if hasattr(self, "result"):
            self.result.set("Your win chance: --     Opponent: --")
            self.bar["value"] = 0
            self.status.set("Decks or model changed. Calculate to update the probability.")

    def _matchup(self):
        return {"player": self.player.get(), "opponent": self.opponent.get()}

    def _set_matchup(self, matchup):
        self.player.set(matchup["player"])
        self.opponent.set(matchup["opponent"])

    def _example(self):
        self._set_matchup(example_matchup())

    def _swap(self):
        matchup = self._matchup()
        self._set_matchup({"player": matchup["opponent"], "opponent": matchup["player"]})

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Choose a trained model", initialdir=DEFAULT_MODEL.parent,
                                          filetypes=[("Keras H5 model", "*.h5"), ("All files", "*.*")])
        if path:
            self.model_path.set(path)

    def _save(self):
        matchup = self._matchup()
        try:
            encode_matchup(matchup)
        except ValueError as error:
            messagebox.showerror("Check your decks", str(error))
            return
        path = filedialog.asksaveasfilename(title="Save matchup", defaultextension=".json",
                                            filetypes=[("Matchup JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(matchup, indent=2) + "\n", encoding="utf-8")
                self.status.set(f"Saved matchup to {path}")
            except OSError as error:
                messagebox.showerror("Could not save matchup", str(error))

    def _load(self):
        path = filedialog.askopenfilename(title="Load matchup", filetypes=[("Matchup JSON", "*.json")])
        if path:
            try:
                matchup = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                encode_matchup(matchup)
                self._set_matchup(matchup)
            except (OSError, ValueError) as error:
                messagebox.showerror("Could not load matchup", str(error))

    def _predict(self):
        matchup = self._matchup()
        try:
            encode_matchup(matchup)
            if not self.model_path.get().strip():
                raise ValueError("Choose a trained model file.")
        except ValueError as error:
            messagebox.showerror("Check your decks", str(error))
            return
        self.pending_revision = self.revision
        self.predict_button.configure(state="disabled")
        self.result.set("Calculating...")
        self.bar["value"] = 0
        self.status.set("Loading the model and predicting... The first calculation can take a few seconds.")
        self.future = self.executor.submit(self.predictor.predict, deepcopy(matchup), self.model_path.get())
        self.root.after(100, self._poll_prediction)

    def _poll_prediction(self):
        if not self.future.done():
            self.root.after(100, self._poll_prediction)
            return
        self.predict_button.configure(state="normal")
        try:
            probabilities = self.future.result()
        except Exception as error:
            self.result.set("Prediction unavailable")
            self.status.set("Check the model file and your Python environment, then try again.")
            messagebox.showerror("Prediction failed", str(error))
            return
        if self.pending_revision != self.revision:
            self.status.set("Decks or model changed during prediction. Calculate again for the current matchup.")
            return
        player = probabilities["player"] * 100
        opponent = probabilities["opponent"] * 100
        self.result.set(f"Your win chance: {player:.1f}%     Opponent: {opponent:.1f}%")
        self.bar["value"] = player
        self.status.set(f"Prediction complete using {Path(self.model_path.get()).name}.")

    def _close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main():
    root = tk.Tk()
    DeckPredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
