"""Process card features, interactions within each deck, then the matchup."""
import keras
from keras.layers import (
    Dense, Flatten, Dropout, GlobalAveragePooling1D, EinsumDense,
    MultiHeadAttention, Lambda, Add, Multiply, Concatenate, LayerNormalization
)
from keras.optimizers import Adam
import tensorflow as tf
CARD_FEATURES = 32
MATCHUP_LAYERS = 3


def build_model(input_shape, *, matchup_layers=MATCHUP_LAYERS, card_features=CARD_FEATURES):
    # One deck has shape (cards, features). Both players use this same encoder.
    deck = keras.Input(shape=input_shape[1:])
    # Uncomment these and the two mask applications below to ignore absent cards.
    # present = Lambda(lambda x: x[:, :, :1], name='card_presence')(deck)
    # attention_mask = Lambda(
    #     lambda x: x[:, None, :, 0] > 0, name='attention_mask'
    # )(deck)

    # Each card/tower row has its own projection, applied in one batched operation.
    x = EinsumDense(
        'bci,cio->bco',
        output_shape=(input_shape[1], card_features),
        bias_axes='co',
        activation='relu',
        name='card_features',
    )(deck)

    # Sort row indices so the nine present rows come first.
    indices = Lambda(
        lambda t: tf.argsort(
            t[:, :, 0],                     # Original presence values.
            axis=1,
            direction="DESCENDING",
            stable=True,
        )[:, :9],
        name='present_indices',
    )(deck)                                  # (batch, 9)

    # Collect those transformed rows independently for each example.
    x = Lambda(
        lambda tensors: tf.gather(
            tensors[0],                     # Transformed card vectors.
            tensors[1],                     # Selected row indices.
            axis=1,
            batch_dims=1,
        ),
        name='collect_present_cards',
    )([x, indices])                          # (batch, 9, CARD_FEATURES)

    # transformer block 1
    for _ in range(2):
        attention = MultiHeadAttention(num_heads=2, key_dim=card_features // 2)(x, x,)
        x = Add()([x, attention])
        
        x = LayerNormalization()(x)
        snn = Dense(64, activation="relu")(x)
        snn = Dense(card_features)(snn)
        x = Add()([x, snn])
        x = LayerNormalization()(x)


    x = Dense(card_features - 1, name='card_content')(x) # neskor pridam este ktory je to hrac
    deck_encoder = keras.Model(deck, x, name='deck_encoder')

    match = keras.Input(shape=input_shape)
    

    #whole match

    player_1 = Lambda(lambda x: x[:, 0], name='player_1')(match)
    player_1 = deck_encoder(player_1)
    player_1 = Lambda(lambda t: tf.concat(
        [t, tf.ones_like(t[:, :, :1])], axis=-1
    ))(player_1)

    player_2 = Lambda(lambda x: x[:, 1], name='player_2')(match)
    player_2 = deck_encoder(player_2)
    player_2 = Lambda(lambda t: tf.concat(
        [t, tf.zeros_like(t[:, :, :1])], axis=-1
    ))(player_2)

    cls_ids = Lambda(
        lambda t: tf.zeros_like(t[:, :1, 0], dtype=tf.int32),
        name='cls_ids',
    )(player_1)
    cls = keras.layers.Embedding(1, card_features)(cls_ids)

    x = Concatenate(axis=1)([cls, player_1, player_2])

    #transformer block
    for _ in range(matchup_layers):
        attention = MultiHeadAttention(num_heads=4, key_dim=card_features // 2)(x, x)
        x=Add()([x, attention])
        x = LayerNormalization()(x)
        snn = Dense(64, activation="relu")(x)
        snn = Dense(card_features)(snn)
        x = Add()([x, snn])
        x = LayerNormalization()(x)

    # x = Flatten()(x)
    # Calling the same encoder twice shares all card and attention weights.
    # x = Concatenate()([deck_encoder(player_1), deck_encoder(player_2)])
    # x = Dense(128, activation='relu', name='matchup')(x)
    # x = Dropout(0.3)(x)
    # x = Dense(128, activation="relu")(x)


    # separat polling for players
    # First nine rows belong to player 1; last nine to player 2.
    # p1 = Lambda(lambda t: t[:, :9, :])(x)
    # p2 = Lambda(lambda t: t[:, 9:, :])(x)

    # p1 = GlobalAveragePooling1D()(p1)  # (batch, CARD_FEATURES)
    # p2 = GlobalAveragePooling1D()(p2)  # (batch, CARD_FEATURES)
    # summary = Concatenate()([p1, p2])  # (batch, 32)

    # common polling for players
    # summary = GlobalAveragePooling1D()(x)  # (batch, CARD_FEATURES)

    #csv
    summary = Lambda(lambda t: t[:, 0, :])(x)

    prediction = Dense(2, activation="softmax")(summary)
    # prediction = Dense(2, activation='softmax')(x)
    cnn = keras.Model(match, prediction)
    cnn.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
    return cnn
