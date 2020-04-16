import tensorflow as tf
import numpy as np


def get_fully_connected(**kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(kwargs.get('vocab_size'), kwargs.get('embedding_size')),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1)])


def get_lstm(**kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(kwargs.get('vocab_size'), kwargs.get('embedding_size')),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(kwargs.get('lstm_cells', 16)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)])


def get_bilstm(**kwargs):
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(kwargs.get('vocab_size'), kwargs.get('embedding_size')),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(kwargs.get('lstm_cells', 16))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)])


def get_sos_model(**kwargs):
    sent_len = kwargs.get('sent_len')
    embed_size = kwargs.get('embedding_size')
    seq_len = kwargs.get("seq_len")
    assert seq_len % sent_len == 0, "sequence length must be a multiple of sentence length"
    sent_per_obs = seq_len // sent_len

    lstm_units_1 = 16
    lstm_units_2 = kwargs.get('lstm_cells', 16)

    inputs = tf.keras.layers.Input(shape=(None,))
    embedded = tf.keras.layers.Embedding(kwargs.get('vocab_size'), embed_size)(inputs)
    reshaped = tf.reshape(embedded, (-1, sent_len, embed_size))
    lstm_level1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1))(reshaped)
    reshaped_level2 = tf.reshape(lstm_level1, (-1, sent_per_obs, lstm_units_1*2))
    lstm_level2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2))(reshaped_level2)
    classifier = tf.keras.layers.Dense(1)(lstm_level2)

    model = tf.keras.Model(inputs=inputs, outputs=classifier)
    return model


def get_sos_score(**kwargs):
    """
    softmax at sentence level, then average.
    :param kwargs:
    :return: keras model
    """
    sent_len = kwargs.get('sent_len')
    embed_size = kwargs.get('embedding_size')
    seq_len = kwargs.get("seq_len")
    assert seq_len % sent_len == 0, "sequence length must be a multiple of sentence length"
    sent_per_obs = seq_len // sent_len

    lstm_units_1 = 16

    inputs = tf.keras.layers.Input(shape=(None,))
    embedded = tf.keras.layers.Embedding(kwargs.get('vocab_size'), embed_size)(inputs)
    reshaped = tf.reshape(embedded, (-1, sent_len, embed_size))
    lstm_level1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1))(reshaped)
    score = tf.keras.layers.Dense(1, activation='softmax')(lstm_level1)
    reshaped_level2 = tf.reshape(score, (-1, sent_per_obs))

    classifier = tf.reshape(tf.math.reduce_mean(reshaped_level2, 1), (-1, 1))

    model = tf.keras.Model(inputs=inputs, outputs=classifier)
    return model


def get_learned_scores(**kwargs):
    """
    "scores" each sentence, then multiply by score before next squence layer
    :param kwargs:
    :return: keras model
    """
    sent_len = kwargs.get('sent_len')
    embed_size = kwargs.get('embedding_size')
    seq_len = kwargs.get("seq_len")
    assert seq_len % sent_len == 0, "sequence length must be a multiple of sentence length"
    sent_per_obs = seq_len // sent_len

    lstm_units_1 = 16
    lstm_units_2 = kwargs.get('lstm_cells', 16)

    inputs = tf.keras.layers.Input(shape=(None,))
    embedded = tf.keras.layers.Embedding(kwargs.get('vocab_size'), embed_size)(inputs)
    reshaped = tf.reshape(embedded, (-1, sent_len, embed_size))
    lstm_level1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1))(reshaped)

    score = tf.keras.layers.Dense(1, activation='softmax')(lstm_level1)
    weighted_reshaped = tf.math.multiply(lstm_level1, score)
    reshaped_level2 = tf.reshape(weighted_reshaped, (-1, sent_per_obs, lstm_units_1*2))

    lstm_level2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2))(reshaped_level2)
    classifier = tf.keras.layers.Dense(1)(lstm_level2)

    model = tf.keras.Model(inputs=inputs, outputs=classifier)
    return model


model_map = {
    'fully_connected': get_fully_connected,
    'lstm': get_lstm,
    'bilstm': get_bilstm,
    'sos': get_sos_model,
    'score': get_sos_score,
    'l_score': get_learned_scores,
}

if __name__ == '__main__':
    my_data = np.array([list(range(10)), list(reversed(range(10)))])

    score_model = get_learned_scores(sent_len=5, vocab_size=10, embedding_size=2, seq_len=10)
    print(score_model(my_data))

    score_model = get_sos_score(sent_len=5, vocab_size=10, embedding_size=2, seq_len=10)
    print(score_model(my_data))

    my_model = get_sos_model(sent_len=5, vocab_size=10, embedding_size=2, seq_len=10)
    print(my_model(my_data))
    another_model = get_bilstm(vocab_size=10, embedding_size=2,)
    print(another_model(my_data))

