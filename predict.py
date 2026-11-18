"""Encode custom decks and run the saved model without rebuilding or training it."""
import os
from pathlib import Path
import sys

import numpy as np

from data import MAX_LEVEL, card_names, input_names, tower_names

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "outputs" / "model.h5"


def _level(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label}: enter a whole-number level from 1 to {MAX_LEVEL}.")
    try:
        level = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: enter a level from 1 to {MAX_LEVEL}.") from None
    if not 1 <= level <= MAX_LEVEL:
        raise ValueError(f"{label}: level must be from 1 to {MAX_LEVEL}.")
    return level


def encode_deck(deck, label="Deck"):
    """Use exactly the fixed rows, features and float16 precision of data.py."""
    if not isinstance(deck, dict):
        raise ValueError(f"{label}: expected a deck object.")
    cards = deck.get("cards")
    if not isinstance(cards, list) or len(cards) != 8:
        raise ValueError(f"{label}: select exactly eight cards.")
    encoded = np.zeros((len(input_names), 4), dtype=np.float16)
    seen = set()
    for number, card in enumerate(cards, 1):
        if not isinstance(card, dict):
            raise ValueError(f"{label}, card {number}: expected a card object.")
        name = card.get("name")
        if name not in card_names:
            raise ValueError(f"{label}, card {number}: select a card from the list.")
        if name in seen:
            raise ValueError(f"{label}: {name} is selected more than once.")
        seen.add(name)
        flags = [card.get("evolution", False), card.get("hero", False)]
        if any(not isinstance(flag, (bool, int)) or flag not in (0, 1) for flag in flags):
            raise ValueError(f"{label}, {name}: Evolution and Hero must be on or off.")
        level = _level(card.get("level"), f"{label}, {name}")
        encoded[card_names.index(name)] = [1, *flags, level / MAX_LEVEL]

    tower = deck.get("tower")
    if not isinstance(tower, dict) or tower.get("name") not in tower_names:
        raise ValueError(f"{label}: select a tower troop from the list.")
    index = len(card_names) + tower_names.index(tower["name"])
    encoded[index] = [1, 0, 0, _level(tower.get("level"), f"{label}, tower") / MAX_LEVEL]
    return encoded


def encode_matchup(matchup):
    if not isinstance(matchup, dict):
        raise ValueError("Expected a matchup containing player and opponent decks.")
    return np.stack([
        encode_deck(matchup.get("player"), "Your deck"),
        encode_deck(matchup.get("opponent"), "Opponent's deck"),
    ])[None, ...]


class MatchupPredictor:
    """Load once per checkpoint version; keep TensorFlow imports out of the UI."""

    def __init__(self):
        self.model = None
        self.checkpoint = None

    def predict(self, matchup, model_path=DEFAULT_MODEL):
        inputs = encode_matchup(matchup)
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Model not found: {path}\nTrain a model first or choose a saved model.")
        checkpoint = (path, path.stat().st_mtime_ns, path.stat().st_size)
        if checkpoint != self.checkpoint:
            # Configure CUDA before importing TensorFlow, but allow CPU inference too.
            cuda_bin = str(Path(sys.prefix) / "Library" / "bin")
            if cuda_bin not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
            os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import keras
            import tensorflow as tf
            # Old H5 Lambda functions refer to globals in the original model module.
            import model as model_module

            loaded = keras.models.load_model(
                str(path), compile=False, custom_objects={"tf": tf, "model": model_module}
            )
            if loaded.input_shape != (None, 2, len(input_names), 4) or loaded.output_shape != (None, 2):
                raise ValueError(
                    f"Incompatible model: input {loaded.input_shape}, output {loaded.output_shape}.\n"
                    f"Expected (None, 2, {len(input_names)}, 4) and (None, 2)."
                )
            self.model = loaded
            self.checkpoint = checkpoint

        probabilities = np.asarray(self.model(inputs, training=False))[0]
        if (probabilities.shape != (2,) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()
                or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
            raise ValueError("The model did not return two valid win probabilities.")
        # train.py encodes 1 = player 1 wins, 0 = player 2 wins.
        return {"player": float(probabilities[1]), "opponent": float(probabilities[0])}
