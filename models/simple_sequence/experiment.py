import pickle
from pathlib import Path

import tensorflow as tf

from config.config import TB_LOGS, MODEL_DIR, MODEL_CONF
from models.simple_sequence.keras_models import model_map
from preprocessing import load_or_fit_tokenizer, get_dataset, get_padded_sequences, load_tokenizer


class Experiment:
    def __init__(self, name, model_name='fully_connected', batch_size=256, seq_len=200, vocab_size=10000,
                 epochs=100, embedding_size=16, split_sentences=False, sent_len=20, **kwargs):

        self.path = Path(MODEL_DIR) / name
        self.tokenizer = None
        self.name = name

        if self.path.exists():
            """ Loading model when already exists"""
            with open(self.path / MODEL_CONF, 'rb') as f:
                kwargs = pickle.load(f)
            self.tokenizer = load_tokenizer(self.path / 'tokenizer.json')  # refactor this
            self.keras_model = model_map[kwargs['model_name']](**kwargs)
            latest = tf.train.latest_checkpoint(str(self.path))
            self.keras_model.load_weights(latest)
        else:
            """ otherwise creating model and files"""
            kwargs = locals().copy()
            del kwargs['self']
            self.keras_model = model_map[model_name](**kwargs)
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self.path / MODEL_CONF, 'wb') as f:
                pickle.dump(kwargs, f)

        self.config = kwargs
        self.compile_model()

    def compile_model(self):
        self.keras_model.summary()
        self.keras_model.compile(optimizer='adam',
                                 loss=tf.losses.BinaryCrossentropy(from_logits=True),
                                 metrics=['accuracy'],
                                 )

    def _get_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(Path(TB_LOGS, self.name)),
                                                     histogram_freq=10)

        save_model = tf.keras.callbacks.ModelCheckpoint(str(self.path / 'checkpoint.ckpt'),
                                                        save_best_only=True,
                                                        monitor='val_accuracy',
                                                        save_weights_only=True)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=15)
        return [tensorboard, save_model, early_stop]

    def train(self, x_train, x_val, x_test, y_train, y_val, y_test, epochs=None):

        if epochs is None:
            epochs = self.config['epochs']
        texts = [x_train, x_val, x_test]
        labels = [y_train, y_val, y_test]

        self.tokenizer = load_or_fit_tokenizer(self.path, self.config['vocab_size'], corpus=x_train)
        padded_sequences = [get_padded_sequences(t, self.tokenizer,
                                                 seq_len=self.config['seq_len'],
                                                 split_sentences=self.config['split_sentences'],
                                                 sent_len=self.config['sent_len'])[0] for t in texts]
        datasets = [get_dataset((x, y), batch_size=self.config['batch_size']) for x, y in zip(padded_sequences, labels)]

        _ = self.keras_model.fit(datasets[0],
                                 validation_data=datasets[1],
                                 epochs=epochs,
                                 callbacks=self._get_callbacks())

        _ = self.keras_model.evaluate(datasets[2], verbose=2)

    def predict(self, predict_data):
        padded_seq, _ = get_padded_sequences(predict_data, self.tokenizer, **self.config)
        return self.keras_model.predict(padded_seq)
