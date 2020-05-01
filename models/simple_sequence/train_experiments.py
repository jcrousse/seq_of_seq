from data_reader import DataReader
from models.simple_sequence.experiment import Experiment
from sklearn.model_selection import train_test_split

# todo now:
#  Keep best model on multiple outputs: Concatenate outputs and labels.  --DONE--
#  Split output for well classified and missclassified. Or name them by distance from actual.
#  quantitatively estimate spread of relevance, and show extreme examples.
#  --- New project structure: ---
#  1. Identify sentence relevance in IMDB Dataset --DONE--
#  (2. optional) Generate a dataset of random sentences from a vocabulary with a set of pre-defined
#  sentences appearing in positive examples only. Confirm those pre-defined sentences are seen as relevant
#  with current model
#  3. Apply same logic to paragraph relevance: IMDB paragraphs cs random paragraphs. Generate random datasets.
#  split by paragraph instead of sentences so it is easier.

n_test = 5000
n_train = 15000
n_val = 2000


test_texts, test_labels = DataReader(0.5, dataset='test').take(n_test)

train_dr = DataReader(0.5)
train_texts, train_labels = train_dr.take(n_train + n_val)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=n_val)

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
    # 'l_score200_20': {'model_name': 'l_score', 'split_sentences': True},
    # "testus": {'model_name': 'l_score', 'split_sentences': True},
    # 'l_concat300_15': {'model_name': 'l_score', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15, 'epochs': 50,
    #                    'concat_outputs': True}
    'score200_20': {'model_name': 'score', 'split_sentences': True},
}

for experiment in experiments:
    name, config = experiment, experiments[experiment]
    model = Experiment(name, overwrite=True, **config)
    model.train(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
