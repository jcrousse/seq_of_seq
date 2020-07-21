import os

DIR_LABEL_MAP = {
    'pos': 1,
    'neg': 0
}

TB_LOGS = 'tensorboard_logs'
MODEL_DIR = 'saved_models'
MODEL_CONF = 'config.pkl'
PARAGRAPH_SEPARATOR = " endofparagraph "

EXP_DIRS = ["train", "test"]

FLOYDHUB = "local"
if not os.path.isfile('.iamlocal'):
    FLOYDHUB = "remote"
elif os.path.isfile('floydwks'):
    FLOYDHUB = "workspace"

if FLOYDHUB == 'local':
    PKL_LOCATION = '/media/john/Johnny-nomad/Work/sos/bert_dataset'
    TXT_LOCATION = 'data/aclImdb_v1'
elif FLOYDHUB == "remote":
    PKL_LOCATION = 'imdb_bert/1'
    TXT_LOCATION = '/floyd/input/imdb_txt'
elif FLOYDHUB == "workspace":
    PKL_LOCATION = '/floyd/input/imdb_bert1'
    TXT_LOCATION = '/floyd/input/imdb_txt_sets'
