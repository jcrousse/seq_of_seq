import sys
import getopt

import wandb

from data_interface import get_dataset
from models.simple_sequence.experiment import Experiment

# todo now:
#   - Ensure working properly with BERT pre-processing (as embedding param),
#   - Same but on FH

# Always applicable: Split sentences or split paragraphs (depends on model)

# train_examples = 15000
# N_VAL = 4000
# sent_len = 15
# num_sent = 20
# lstm_units_1 = 16
# vocab_size = 15000
PRE_CALC_EMBEDD = False
dataset = "P4_from1200_vocab200_fromPNone_noextra"  # "P4_from1200_vocab200_fromPNone_noextra" "aclImdb"
model_type = "combined"  # attention, sos, combined
# embedding_size = 16
# batch_size = 128

hparams = {
    'model_name': 'l_score',
    'train_examples': 15000,
    'n_val': 4000,
    'sent_len': 15,
    'num_sent': 20,
    'lstm_units_1': 16,
    'vocab_size': 15000,
    'embedding_size': 16,
    'batch_size': 128,
    'model_type': "attention",
    "split_sentences": "paragraph"
}
wandb.init(config=hparams)
config = wandb.config


N_TEST = 400
experiment_name = "test_wandb"
wandb.init(project="sos")

# model_type_map = {
#     0: "attention",
#     1: "sos",
#     2: "combined"
# }

if __name__ == '__main__':
    long_options = ["sent_len:num_sent:vocab_size:batch_size:lstm_units_1:dataset:examples:model:word_embeddings:"]
    short_options = ["s:n:v:b:l:d:e:m:w:"]
    args = sys.argv[1:]
    try:
        arguments, values = getopt.getopt(args, short_options, long_options)
    except getopt.error as err:
        print(f"train.py -s <sent_len>")
    else:
        for arg, val, current_value in arguments:
            if arg in ("-s", "--sent_len"):
                hparams['sent_len'] = int(val)
            elif arg in ("-n", "--num_sent"):
                hparams['num_sent'] = int(val)
            elif arg in ("-v", "--vocab_size"):
                hparams['vocab_size'] = int(val)
            elif arg in ("-b", "--batch_size"):
                hparams['batch_size'] = int(val)
            elif arg in ("-l", "--lstm_units_1"):
                hparams['lstm_units_1'] = int(val)
            elif arg in ("-d", "--dataset"):
                hparams['dataset'] = int(val)
            elif arg in ("-e", "--examples"):
                hparams['train_examples'] = int(val)
            # elif arg in ("-m", "--model"):
            #     hparams['model_type'] = model_type_map[int(val)]
            elif arg in ("-w", "--word_embeddings"):
                # change here for bert embeddings
                hparams['embedding_size'] = int(val)

        wandb.config.update(hparams)
    print(config)
    for k in hparams.keys():
        hparams[k] = config[k]
    print(hparams)
    datasets = get_dataset(dataset, config['train_examples'], config['n_val'], N_TEST, PRE_CALC_EMBEDD)

    model = Experiment(experiment_name, overwrite=True, **hparams)
    model.train(datasets)
