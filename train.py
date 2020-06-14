import sys
import getopt

import wandb

from data_interface import get_dataset
from models.simple_sequence.experiment import Experiment

# todo now:
#   - Ensure working properly with BERT pre-processing (as embedding param),
#   - Same but on FH

# Always applicable: Split sentences or split paragraphs (depends on model)

train_examples = 5000
N_VAL = 4000
sent_len = 15
num_sent = 20
lstm_units_1 = 16
vocab_size = 10000
PRE_CALC_EMBEDD = False
dataset = "aclImdb"  # "P4_from1200_vocab200_fromPNone_noextra"
model_type = "combined"  # attention, sos, combined
embedding_size = 16

N_TEST = 400
experiment_name = "test_wandb"
wandb.init(project="sos", sync_tensorboard=True)

model_type_map = {
    0: "attention",
    1: "sos",
    2: "combined"
}

if __name__ == '__main__':
    long_options = ["sent_len:num_sent:vocab_size:batch_size:lstm_units_1:dataset:examples:model:word_embeddings:"]
    short_options = ["s:n:v:b:l:d:e:m:w:"]
    args = sys.argv[1:]
    try:
        arguments, values = getopt.getopt(args, short_options, long_options)
    except getopt.error as err:
        print(f"train.py -i <sent_len>")
    else:
        for arg, val, current_value in arguments:
            if arg in ("-s", "--sent_len"):
                sent_len = int(val)
            elif arg in ("-n", "--num_sent"):
                num_sent = int(val)
            elif arg in ("-v", "--vocab_size"):
                vocab_size = int(val)
            elif arg in ("-l", "--lstm_units_1"):
                lstm_units_1 = int(val)
            elif arg in ("-d", "--dataset"):
                dataset = int(val)
            elif arg in ("-e", "--examples"):
                train_examples = int(val)
            elif arg in ("-m", "--model"):
                model_type = model_type_map[int(val)]
            elif arg in ("-w", "--word_embeddings"):
                # change here for bert embeddings
                embedding_size = int(val)
    model_config = {
        'epochs': 20, 'concat_outputs': True, 'model_name': "l_score", 'sent_len': sent_len, 'num_sent': num_sent,
        'vocab_size': vocab_size, 'lstm_units_1': lstm_units_1, 'model_type': model_type}
    datasets = get_dataset(dataset, train_examples, N_VAL, N_TEST, PRE_CALC_EMBEDD)

    model = Experiment(experiment_name, overwrite=True, **model_config)
    model.train(datasets)
