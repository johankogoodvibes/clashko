"""A logistic winner model using differences between pre-match deck features."""

import keras
import tensorflow as tf

from data import MAX_LEVEL, card_names


@tf.keras.utils.register_keras_serializable(package="clash_royale")
class MatchupDifferences(keras.layers.Layer):
    """Compare card and tower presence and deck-level summaries."""

    def call(self, match):
        cards = match[:, :, :len(card_names), :]
        present = cards[..., 0] > 0
        levels = cards[..., 3] * MAX_LEVEL
        average_level = tf.math.divide_no_nan(
            tf.reduce_sum(levels, axis=-1), tf.reduce_sum(cards[..., 0], axis=-1)
        )
        minimum_level = tf.reduce_min(tf.where(present, levels, MAX_LEVEL), axis=-1)
        maxed_cards = tf.reduce_sum(tf.cast(present & (levels == MAX_LEVEL), match.dtype), axis=-1)
        low_level_cards = tf.reduce_sum(tf.cast(present & (levels <= MAX_LEVEL - 2), match.dtype), axis=-1)
        evolutions = tf.reduce_sum(cards[..., 1], axis=-1)
        heroes = tf.reduce_sum(cards[..., 2], axis=-1)

        summaries = tf.stack(
            [average_level, minimum_level, maxed_cards, low_level_cards, evolutions, heroes], axis=-1
        )
        deck_features = tf.concat([match[..., 0], summaries], axis=-1)
        tower_levels = tf.reduce_sum(match[:, :, len(card_names):, 3], axis=-1) * MAX_LEVEL
        tower_gap = tf.where(
            tf.reduce_all(tower_levels > 0, axis=1), tower_levels[:, 0] - tower_levels[:, 1], 0.0
        )
        return tf.concat([deck_features[:, 0] - deck_features[:, 1], tower_gap[:, None]], axis=-1)


def build_logistic_model(input_shape):
    match = keras.Input(shape=input_shape)
    differences = MatchupDifferences(name="matchup_differences")(match)
    # Without a bias, swapping sides negates the log-odds and exchanges the probabilities.
    log_odds = keras.layers.Dense(1, use_bias=False, kernel_initializer="zeros", name="win_log_odds")(differences)
    logits = keras.layers.Concatenate()([keras.layers.Rescaling(0.0)(log_odds), log_odds])
    winner = keras.layers.Softmax(name="winner")(logits)
    model = keras.Model(match, winner, name="logistic_winner")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={"winner": "categorical_crossentropy"},
        metrics={"winner": ["accuracy"]},
    )
    return model
