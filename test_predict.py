"""Run with .venv\python.exe -m unittest test_predict -v."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import numpy as np

from data import MAX_LEVEL, card_names, input_names, tower_names
from deck_predictor import example_matchup
from predict import MatchupPredictor, encode_matchup


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
        np.testing.assert_array_equal(encoded[0, 0, card_names.index("Firecracker")], [1, 1, 0, 14/MAX_LEVEL])
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
