import tensorflow as tf

from data_reader import DataReader
from models.simple_sequence.keras_models import model_map
from preprocessing import split_sentences as sent_splitter


# todo now:
#  - sequence of sequence, yay \o/
#  start by generating junk in text and ruining it?  Or start by Making SoS work to review attention?
#  OR, try to get it to work with seq 500 by reproducing this:
#  https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

n_test = 10000
n_train = 5000
n_val = 2000


test_texts, test_labels = DataReader(0.5, dataset='test').take(n_test)

train_dr = DataReader(0.5)
train_texts, train_labels = train_dr.take(n_train)
val_texts, val_labels = train_dr.take(n_val)


def get_dataset(texts, labels, tokenizer, batch_size=256, seq_len=200):
    tokens = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='post', maxlen=seq_len)
    dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
    batches = (dataset.shuffle(1000).padded_batch(batch_size, padded_shapes=([None], [])))
    return batches

def sentence_processing(text_list, tokenizer):
    pass


def run_experiment(experiment_name, model_name='fully_connected', batch_size=256, seq_len=200, vocab_size=10000,
                   epochs=100, embedding_size=16, split_sentences=False, **kwargs):
    """
    :param experiment_name: string to name the experiment. Will be used as a folder to store tensorborad output
    :param model_name: key in the model_map dictionary to get the keras model from
    :param batch_size: int batch size (default 256)
    :param seq_len: maximal sequence length (cropped)
    :param vocab_size: number of words in vocabulary
    :param epochs: number of epochs to train on
    :param embedding_size: length of word embeddings
    :param split_sentences: whether to split text by sentences & pad them
    :return: None
    """
    kwargs = locals().copy()
    model = model_map[model_name](**kwargs)

    tokenizer_train = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                            filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer_train.fit_on_texts(train_texts)
    if split_sentences:
        train_texts = [sentence_processing(text) for text in train_texts]

    train_batches = get_dataset(train_texts, train_labels, tokenizer_train, batch_size=batch_size, seq_len=seq_len)
    validation_batches = get_dataset(val_texts, val_labels, tokenizer_train, batch_size=batch_size, seq_len=seq_len)
    test_batches = get_dataset(test_texts, test_labels, tokenizer_train, batch_size=batch_size, seq_len=seq_len)

    model.summary()

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard_logs/' + str(experiment_name),
                                                          histogram_freq=10)

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  )

    _ = model.fit(train_batches,
                  validation_data=validation_batches,
                  epochs=epochs,
                  callbacks=[tensorboard_callback])

    results = model.evaluate(test_batches, verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))


experiments = {
    # 'fc_200': {'seq_len': 200},
    # 'fc_500': {'seq_len': 500},
    # 'lstm_200': {'model_name': 'lstm', 'epochs': 20},
    # 'lstm_300': {'model_name': 'lstm', 'seq_len': 300, 'epochs': 100},
    # 'lstm_400': {'model_name': 'lstm', 'seq_len': 400, 'epochs': 150},
    # 'lstm_500': {'model_name': 'lstm', 'seq_len': 500, 'epochs': 150},
    # 'bilstm_200': {'model_name': 'bilstm', 'epochs': 100},
    # 'bilstm_300': {'model_name': 'bilstm', 'seq_len': 300, 'epochs': 100},
    # 'bilstm_400': {'model_name': 'bilstm', 'seq_len': 400, 'epochs': 150},
    # 'bilstm_500': {'model_name': 'bilstm', 'seq_len': 500, 'epochs': 150},
    # 'bilstm_1000': {'model_name': 'bilstm', 'seq_len': 1000, 'epochs': 150},
    # 'bilstm_2000': {'model_name': 'bilstm', 'seq_len': 2000, 'epochs': 150},
    # 'lstm_1000': {'model_name': 'lstm', 'seq_len': 1000, 'epochs': 80},
    # 'fc_1000': {'seq_len': 500, 'epochs': 200},
    'sos': {'split_sentences': True}
}

for experiment in experiments:
    run_experiment(experiment, **experiments[experiment])
