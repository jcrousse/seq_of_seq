from data_reader import DataReader
from models.simple_sequence.experiment import Experiment

# todo now:
#  Try different attention approaches:
#    -Softmax
#    -Softmax + weighted sum
#    -Double output
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
    "testus": {'model_name': 'l_score', 'split_sentences': True}
}

for experiment in experiments:
    name, config = experiment, experiments[experiment]
    model = Experiment(name, overwrite=True, **config)
    model.train(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels)
