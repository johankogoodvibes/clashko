"""Run with py -3.10 -m unittest test_predict -v."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock
import tkinter as tk

import numpy as np

from data import MAX_LEVEL, card_names, input_names, tower_names
from deck_predictor import CardCombobox, example_matchup
from predict import MatchupPredictor, encode_matchup


class AutocompleteFocusTests(unittest.TestCase):
    def test_focus_check_handles_native_popdown_without_python_wrapper(self):
        # Use Tcl without opening a window or changing desktop focus.
        interpreter = tk.Tcl()
        root = tk.Misc()
        root.master = None
        combo = CardCombobox.__new__(CardCombobox)
        combo.tk = interpreter
        combo.master = root
        combo._w = ".cards"
        combo.children = {}
        root.children = {"cards": combo}
        combo._hide = Mock()
        interpreter.createcommand("winfo", lambda command, path: 1)

        for focused in (".cards", ".cards.popdown.f.l", ".other", ""):
            with self.subTest(focused=focused):
                interpreter.createcommand("focus", lambda: focused)
                combo._hide.reset_mock()
                if focused == ".cards.popdown.f.l":
                    with self.assertRaisesRegex(KeyError, "popdown"):
                        combo.focus_get()
                combo._check_focus()
                self.assertEqual(combo._hide.called, focused != ".cards")

        interpreter.createcommand("winfo", lambda command, path: 0)
        combo._hide.reset_mock()
        combo._check_focus()
        combo._hide.assert_not_called()


class PredictionTests(unittest.TestCase):
    def test_fixed_rows_features_and_sides(self):
        matchup = example_matchup()
        matchup["player"]["cards"][0].update(level=16, hero=True)
        matchup["opponent"]["tower"] = {"name": "Royal Chef", "level": 12}
        encoded = encode_matchup(matchup)
        self.assertEqual(encoded.shape, (1, 2, len(input_names), 4))
        self.assertEqual(encoded.dtype, np.float16)
        np.testing.assert_array_equal(encoded[0, :, :, 0].sum(axis=1), [9, 9])
        np.testing.assert_array_equal(encoded[0, 0, card_names.index("Hog Rider")], [1, 0, 1, 1])
        np.testing.assert_array_equal(encoded[0, 0, card_names.index("Firecracker")], [1, 1, 0, 16/MAX_LEVEL])
        tower = len(card_names) + tower_names.index("Royal Chef")
        np.testing.assert_array_equal(encoded[0, 1, tower], [1, 0, 0, 12/MAX_LEVEL])
        np.testing.assert_array_equal(encoded[0, 0, card_names.index("Giant")], [0, 0, 0, 0])

    def test_card_order_does_not_change_encoding(self):
        matchup = example_matchup()
        reversed_cards = deepcopy(matchup)
        reversed_cards["player"]["cards"].reverse()
        np.testing.assert_array_equal(encode_matchup(matchup), encode_matchup(reversed_cards))

    def test_swap_reverses_only_player_axis(self):
        matchup = example_matchup()
        swapped = {"player": matchup["opponent"], "opponent": matchup["player"]}
        np.testing.assert_array_equal(encode_matchup(matchup)[:, ::-1], encode_matchup(swapped))

    def test_invalid_decks_rejected(self):
        for field, value in (("name", "Unknown"), ("level", 0), ("level", 17),
                             ("level", 14.5), ("level", "14.5"), ("level", True),
                             ("evolution", 2), ("hero", "false")):
            with self.subTest(field=field, value=value):
                matchup = example_matchup()
                matchup["player"]["cards"][0][field] = value
                with self.assertRaises(ValueError):
                    encode_matchup(matchup)
        for bad_deck in (None, {}, {"cards": []}):
            with self.subTest(deck=bad_deck), self.assertRaises(ValueError):
                encode_matchup({"player": bad_deck})
        matchup = example_matchup()
        matchup["player"]["cards"][1] = matchup["player"]["cards"][0]
        with self.assertRaisesRegex(ValueError, "more than once"):
            encode_matchup(matchup)
        matchup = example_matchup()
        matchup["player"]["tower"]["name"] = "Unknown tower"
        with self.assertRaisesRegex(ValueError, "tower troop"):
            encode_matchup(matchup)

    def test_model_output_maps_win_to_class_one(self):
        class FakeModel:
            def __call__(self, inputs, training):
                self.inputs = inputs
                return np.array([[0.25, 0.75]])

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.h5"
            path.touch()
            predictor = MatchupPredictor()
            predictor.model = FakeModel()
            predictor.checkpoint = (path.resolve(), path.stat().st_mtime_ns, path.stat().st_size)
            result = predictor.predict(example_matchup(), path)
            self.assertEqual(result, {"player": 0.75, "opponent": 0.25})
            np.testing.assert_array_equal(predictor.model.inputs, encode_matchup(example_matchup()))

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Model not found"):
                MatchupPredictor().predict(example_matchup(), Path(temporary) / "missing.h5")


if __name__ == "__main__":
    unittest.main()
