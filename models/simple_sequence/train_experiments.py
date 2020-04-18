from pathlib import Path
import pickle

import tensorflow as tf

from data_reader import DataReader
from models.simple_sequence.keras_models import model_map
from preprocessing import load_or_fit_tokenizer, get_dataset, get_padded_sequences
from config.config import TB_LOGS, MODEL_DIR, MODEL_CONF

# todo now:
#  refactor model into a Class with save/load/overwrite (and cleanup directories), to avoid code duplication.
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

    model_path = Path(MODEL_DIR) / experiment_name
    model_path.mkdir(parents=True, exist_ok=True)
    kwargs = locals().copy()
    model = model_map[model_name](**kwargs)
    with open(Path(model_path) / MODEL_CONF, 'wb') as f:
        pickle.dump(kwargs, f)

    texts = [train_texts,  val_texts, test_texts]
    labels = [train_labels, val_labels, test_labels]

    tokenizer = load_or_fit_tokenizer(model_path, vocab_size, corpus=train_texts)

    padded_sequences = [get_padded_sequences(t, tokenizer, seq_len=seq_len, split_sentences=split_sentences,
                                             sent_len=sent_len)[0] for t in texts]
    datasets = [get_dataset((x, y), batch_size=batch_size) for x, y in zip(padded_sequences, labels)]

    model.summary()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(Path(TB_LOGS, experiment_name)),
                                                 histogram_freq=10)
    save_model = tf.keras.callbacks.ModelCheckpoint(str(model_path / 'checkpoint.ckpt'),
                                                    save_best_only=True,
                                                    monitor='val_accuracy',
                                                    save_weights_only=True)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=15)

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
    # 'l_score300_15': {'model_name': 'l_score', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15, 'epochs': 50},
    "testus": {'split_sentences': True}
}

for experiment in experiments:
    run_experiment(experiment, **experiments[experiment])
