from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense


def build_gru_lang_binary(text_vec_layer, vocab_size, embed_dim=128, gru_units=128):
    model = Sequential([
        text_vec_layer,
        Embedding(input_dim=vocab_size, output_dim=embed_dim),
        GRU(units=gru_units),
        Dense(units=1, activation="sigmoid")                        # 1 (positive) or 0 (negative)
    ])
    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics="accuracy")
    return model