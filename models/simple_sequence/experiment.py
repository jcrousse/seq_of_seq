import pickle
from shutil import rmtree
from pathlib import Path

import tensorflow as tf

from config.config import TB_LOGS, MODEL_DIR, MODEL_CONF
from models.simple_sequence.keras_models import model_map
from preprocessing import load_or_fit_tokenizer, get_dataset, get_padded_sequences, load_tokenizer


class Experiment:
    def __init__(self, name, model_name='fully_connected', batch_size=256, seq_len=200, vocab_size=10000,
                 epochs=100, embedding_size=16, split_sentences=False, sent_len=20, overwrite=False, **kwargs):

        self.name = name
        self.path = Path(MODEL_DIR) / name
        self.log_path = Path(TB_LOGS) / self.name
        self.tokenizer = None

        if self.path.exists() and not overwrite:
            """ Loading model when already exists"""
            with open(self.path / MODEL_CONF, 'rb') as f:
                kwargs = pickle.load(f)
            self.tokenizer = load_tokenizer(self.path / 'tokenizer.json')  # refactor this
            self.keras_model = model_map[kwargs['model_name']](**kwargs)
            latest = tf.train.latest_checkpoint(str(self.path))
            self.keras_model.load_weights(latest)
        else:
            """ otherwise creating model and files"""
            for path in [p for p in [self.path, self.log_path] if p.exists()]:
                rmtree(path)
            kwargs = {**locals().copy(), **kwargs}
            del kwargs['self']
            self.keras_model = model_map[model_name](**kwargs)
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self.path / MODEL_CONF, 'wb') as f:
                pickle.dump(kwargs, f)

        self.config = kwargs

        output_layer = self.keras_model.output
        if isinstance(output_layer, list):
            self.out_shape = output_layer[0].shape[1]
        else:
            self.out_shape = output_layer.shape[1]
        self.compile_model()

    def compile_model(self):
        self.keras_model.summary()
        self.keras_model.compile(optimizer='adam',
                                 loss=tf.losses.BinaryCrossentropy(from_logits=True),
                                 metrics=['accuracy'],
                                 )

    def _get_callbacks(self):
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(self.log_path),
                                                     histogram_freq=10)

        monitor = 'val_accuracy' if len(self.keras_model.output_names) == 1 \
            else f'val_{self.keras_model.output_names[0]}_accuracy'
        save_model = tf.keras.callbacks.ModelCheckpoint(str(self.path / 'checkpoint.ckpt'),
                                                        save_best_only=True,
                                                        monitor=monitor,
                                                        save_weights_only=True)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, min_delta=0.01, patience=15)
        return [tensorboard, save_model, early_stop]

    def train(self, datasets, epochs=None):
        if isinstance(datasets[0], tuple):
            prep_data = self._format_mem_data(datasets, epochs)
        else:
            prep_data = datasets
        concat_output = self.config.get("concat_outputs", False)
        n_outputs = len(self.keras_model.layers[-1].input)
        out_shape = self.out_shape
        datasets = [get_dataset(dt, batch_size=self.config['batch_size'],
                                concat_outputs=concat_output,
                                n_outputs=n_outputs,
                                out_shape=out_shape) for dt in prep_data]
        self._train_from_tfdatasets(datasets, epochs)

    def _format_mem_data(self, dataset_items, epochs=None):

        texts = [dataset_items[0][0], dataset_items[1][0], dataset_items[2][0]]
        labels = [dataset_items[0][1], dataset_items[1][1], dataset_items[2][1]]
        # if self.config.get("concat_outputs", False):
        #     n_outputs = len(self.keras_model.layers[-1].input)
        #     labels = [[[e] * n_outputs for e in sublist] for sublist in labels]

        self.tokenizer = load_or_fit_tokenizer(self.path, self.config['vocab_size'], corpus=dataset_items[0][0])
        padded_sequences = [get_padded_sequences(t, self.tokenizer,
                                                 seq_len=self.config['seq_len'],
                                                 split_sentences=self.config['split_sentences'],
                                                 sent_len=self.config['sent_len'])[0] for t in texts]

        # datasets = [get_dataset((
        #     {"input": x},
        #     {"output": y, "output_2": y}),
        #     batch_size=self.config['batch_size'],
        #     out_shape=self.out_shape)
        #             for x, y in zip(padded_sequences, labels)]
        return [({"input": x}, {"output": y, "output_2": y}) for x, y in zip(padded_sequences, labels)]


    def _train_from_tfdatasets(self, dataset_list, epochs=None):
        """
        train from a list of three TF datasets: train, val and test
        """
        if epochs is None:
            epochs = self.config['epochs']

        _ = self.keras_model.fit(dataset_list[0],
                                 validation_data=dataset_list[1],
                                 epochs=epochs,
                                 callbacks=self._get_callbacks())

        _ = self.keras_model.evaluate(dataset_list[2], verbose=2)

    def predict(self, predict_data, return_sentences=False):
        padded_seq, sentences = get_padded_sequences(predict_data, self.tokenizer, **self.config)
        if return_sentences:
            return self.keras_model.predict(padded_seq), sentences
        else:
            return self.keras_model.predict(padded_seq)

    def predict_layer(self, predict_data, layer="relevance_reshaped"):
        padded_seq, _ = get_padded_sequences(predict_data, self.tokenizer, **self.config)
        partial_model = tf.keras.Model(
            inputs=self.keras_model.input,
            outputs=self.keras_model.get_layer(layer).output)
        return partial_model.predict(padded_seq)
