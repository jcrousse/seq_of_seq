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

    inputs = tf.keras.layers.Input(shape=(None,), name="input")
    embedded = tf.keras.layers.Embedding(kwargs.get('vocab_size'), embed_size)(inputs)
    reshaped = tf.reshape(embedded, (-1, sent_len, embed_size))
    lstm_level1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1))(reshaped)
    x = tf.keras.layers.Dense(1, activation=None)(lstm_level1)
    logits = tf.reshape(x, (-1, sent_per_obs))
    score = tf.keras.layers.Softmax(name="score")(logits)
    weighted = tf.multiply(lstm_level1, tf.reshape(score, (-1, 1)))

    reshaped_level2 = tf.reshape(weighted, (-1, sent_per_obs, lstm_units_1*2))
    w_average = tf.keras.layers.GlobalAveragePooling1D()(reshaped_level2)

    classifier = tf.keras.layers.Dense(1, name="output")(w_average)

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

    concat_outputs = kwargs.get("concat_outputs", False)

    lstm_units_1 = 16
    lstm_units_2 = kwargs.get('lstm_cells', 16)

    inputs = tf.keras.layers.Input(shape=(None,), name="input")
    embedded = tf.keras.layers.Embedding(kwargs.get('vocab_size'), embed_size)(inputs)
    reshaped = tf.reshape(embedded, (-1, sent_len, embed_size))
    lstm_level1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1))(reshaped)

    x = tf.keras.layers.Dense(1, activation=None)(lstm_level1)
    logits = tf.reshape(x, (-1, sent_per_obs))
    score = tf.keras.layers.Softmax(name="score")(logits)
    weighted = tf.multiply(lstm_level1, tf.reshape(score, (-1, 1)))

    reshaped_level2 = tf.reshape(weighted, (-1, sent_per_obs, lstm_units_1*2))
    w_average = tf.keras.layers.GlobalAveragePooling1D()(reshaped_level2)

    lstm_level2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2))(reshaped_level2)

    if concat_outputs:
        classifier = tf.keras.layers.Dense(1)(lstm_level2)
        classifier2 = tf.keras.layers.Dense(1)(w_average)
        outputs = tf.keras.layers.concatenate([classifier, classifier2], name="output")
    else:
        classifier = tf.keras.layers.Dense(1, name="output")(lstm_level2)
        classifier2 = tf.keras.layers.Dense(1, name="output_2")(w_average)

        outputs = [classifier, classifier2]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
