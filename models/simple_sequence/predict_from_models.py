from pathlib import Path
import pickle

import tensorflow as tf

from data_reader import DataReader
from preprocessing import load_or_fit_tokenizer, get_padded_sequences
from config.config import MODEL_DIR, MODEL_CONF
from models.simple_sequence.keras_models import model_map

MODEL_LOAD = 'testus'

model_path = Path(MODEL_DIR, MODEL_LOAD)
with open(model_path / MODEL_CONF, 'rb') as f:
    model_config = pickle.load(f)


predict_text, predict_labels = DataReader(0.5, dataset='test').take(100)


tokenizer = load_or_fit_tokenizer(model_path, **model_config)
padded_seq, sentences = get_padded_sequences(predict_text, tokenizer, **model_config)

model = model_map[model_config['model_name']](**model_config)
latest = tf.train.latest_checkpoint(str(model_path))
model.load_weights(latest)

predictions = model.predict(padded_seq)
