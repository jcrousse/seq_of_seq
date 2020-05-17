from data_interface import DataReader
from models.simple_sequence.experiment import Experiment
from sklearn.model_selection import train_test_split

from preprocessing.text_preprocessing import split_paragraphs

# todo now:
#  - Try spacy transformers on paragraphs ?
#       >> Make the model more modulable so it can use own embeddings or not.
#       >> Then feed doc.tensor into model as result of pre-processing?
#               Or prep TF Dataset as Tensors?
#       >> Generate a TFRecord dataset with those lists of tensors? Padded? How to pad?
#  - Run train on floydhub?
#  - "Cheat" with set of positive / negative sentences ?

N_TEST = 500
N_TRAIN = 1700
N_VAL = 200

DATASET = "P4_from1200_vocab200_fromPNone_noextra"

test_texts, test_labels = DataReader(0.5, dataset=DATASET, subset='test').take(N_TEST)

train_dr = DataReader(0.5, dataset=DATASET)
train_texts, train_labels = train_dr.take(N_TRAIN + N_VAL)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=N_VAL)

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
    # 'l_concat300_15': {'model_name': 'l_score', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15, 'epochs': 50,
    #                    'concat_outputs': True}
    # 'score200_20': {'model_name': 'score', 'split_sentences': True},
    # 'l_concat200_20': {'model_name': 'l_score', 'split_sentences': True, 'seq_len': 200, 'sent_len': 20, 'epochs': 50,
    #                    'concat_outputs': True},
    # 'score300_15': {'model_name': 'score', 'split_sentences': True, 'seq_len': 300, 'sent_len': 15, 'epochs': 50,},
    "testus": {'model_name': 'l_score', 'split_sentences': True, 'sent_splitter': split_paragraphs, 'sent_len': 200,
               'seq_len': 1200, 'batch_size': 512, 'concat_outputs': True, 'lstm_units_1': 32},
}

for experiment in experiments:
    name, config = experiment, experiments[experiment]
    model = Experiment(name, overwrite=True, **config)
    model.train(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
