import tensorflow as tf

from data_reader import DataReader
from models.simple_sequence.keras_models import model_map
from preprocessing import split_sentences as sent_splitter


# todo now:
#  Load saved model. Re-factor preprocessing pipeline to re-use for predict, incl. save tokenizer
#  extract score value per sentence from saves model at predict
#  html output
#  Generate junk in text and ruining it?  Or start by Making SoS work to review attention?
#  OR, try to get it to work with seq 500 by reproducing this:
#  https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/

n_test = 10000
n_train = 5000
n_val = 2000


test_texts, test_labels = DataReader(0.5, dataset='test').take(n_test)

train_dr = DataReader(0.5)
train_texts, train_labels = train_dr.take(n_train)
val_texts, val_labels = train_dr.take(n_val)


def get_dataset(texts, labels, tokenizer, batch_size=256, seq_len=200, split_sentence=False, sent_len=20):
    if split_sentence:
        sentences = [sent_splitter(t) for t in texts]

        def sent_padding(s):
            return tf.keras.preprocessing.sequence.pad_sequences(s, padding='post', maxlen=sent_len)
    else:
        sentences = [[t] for t in texts]

        def sent_padding(s):
            return s

    tokenized_sentences = [tokenizer.texts_to_sequences(text) for text in sentences]
    padded_sentences = [sent_padding(tokens) for tokens in tokenized_sentences]
    preprocessed_texts = [[w for sentences in sent_list for w in sentences] for sent_list in padded_sentences]
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(preprocessed_texts, padding='post', maxlen=seq_len)
    dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
    batches = dataset.shuffle(1000).padded_batch(batch_size, padded_shapes=([None], []))
    return batches


def run_experiment(experiment_name, model_name='fully_connected', batch_size=256, seq_len=200, vocab_size=10000,
                   epochs=100, embedding_size=16, split_sentences=False, sent_len=20, **kwargs):
    """
    :param experiment_name: string to name the experiment. Will be used as a folder to store tensorborad output
    :param model_name: key in the model_map dictionary to get the keras model from
    :param batch_size: int batch size (default 256)
    :param seq_len: maximal sequence length (cropped)
    :param vocab_size: number of words in vocabulary
    :param epochs: number of epochs to train on
    :param embedding_size: length of word embeddings
    :param split_sentences: whether to split text by sentences & pad them
    :param sent_len: padded sentence length (if splitting by sentences)
    :return: None
    """
    kwargs = locals().copy()
    model = model_map[model_name](**kwargs)

    tokenizer_train = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                            filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer_train.fit_on_texts(train_texts)

    raw_data = [(train_texts, train_labels),  (val_texts, val_labels), (test_texts, test_labels)]

    datasets = [get_dataset(data[0], data[1], tokenizer_train, batch_size=batch_size, seq_len=seq_len,
                            split_sentence=split_sentences, sent_len=sent_len)
                for data in raw_data]

    model.summary()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='tensorboard_logs/' + str(experiment_name),
                                                 histogram_freq=10)
    save_model = tf.keras.callbacks.ModelCheckpoint('saved_models/' + experiment_name,
                                                    save_best_only=True, monitor='val_accuracy')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=5)

    model.compile(optimizer='adam',
                  loss=tf.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  )

    _ = model.fit(datasets[0],
                  validation_data=datasets[1],
                  epochs=epochs,
                  callbacks=[tensorboard, save_model, early_stop])

    results = model.evaluate(datasets[2], verbose=2)

    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))


experiments = {
    # 'fc_200': {'seq_len': 200},
    # 'fc_500': {'seq_len': 500},
    # 'lstm_200': {'model_name': 'lstm', 'epochs': 20},
    # 'lstm_300': {'model_name': 'lstm', 'seq_len': 300, 'epochs': 100},
    # 'lstm_400': {'model_name': 'lstm', 'seq_len': 400, 'epochs': 150},
    # 'lstm_500': {'model_name': 'lstm', 'seq_len': 500, 'epochs': 150},
    # 'bilstm_300': {'model_name': 'bilstm', 'seq_len': 300, 'epochs': 100},
    # 'bilstm_400': {'model_name': 'bilstm', 'seq_len': 400, 'epochs': 150},
    # 'bilstm_500': {'model_name': 'bilstm', 'seq_len': 500, 'epochs': 150},
    # 'bilstm_1000': {'model_name': 'bilstm', 'seq_len': 1000, 'epochs': 150},
    # 'bilstm_2000': {'model_name': 'bilstm', 'seq_len': 2000, 'epochs': 150},
    # 'lstm_1000': {'model_name': 'lstm', 'seq_len': 1000, 'epochs': 80},
    # 'fc_1000': {'seq_len': 500, 'epochs': 200},
    # 'sos300_15': {'model_name': 'sos', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15},
    # 'bilstm_split_300_15': {'model_name': 'bilstm', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15},
    'l_score300_15': {'model_name': 'l_score', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15, 'epochs': 50},
}

for experiment in experiments:
    run_experiment(experiment, **experiments[experiment])
