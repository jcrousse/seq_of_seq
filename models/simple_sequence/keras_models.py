import tensorflow as tf


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


model_map = {
    'fully_connected': get_fully_connected,
    'lstm': get_lstm,
    'bilstm': get_bilstm
}