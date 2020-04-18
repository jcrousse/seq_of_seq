TRAIN_POS_DIR = 'data/aclImdb_v1/aclImdb/train/pos'
TRAIN_NEG_DIR = 'data/aclImdb_v1/aclImdb/train/neg'

TEST_POS_DIR = 'data/aclImdb_v1/aclImdb/test/pos'
TEST_NEG_DIR = 'data/aclImdb_v1/aclImdb/test/neg'

TRAIN_SET = {
    1: TRAIN_POS_DIR,
    0: TRAIN_NEG_DIR
}

TEST_SET = {
    1: TEST_POS_DIR,
    0: TEST_NEG_DIR
}

TB_LOGS = 'tensorboard_logs'
MODEL_DIR = 'saved_models'
MODEL_CONF = 'config.pkl'