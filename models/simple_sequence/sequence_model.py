import tensorflow as tf

from data_reader import DataReader

# todo now:
#  - Tensorboard output.
#  - Comparison with GRU / LSTM  (for model in models, fit & eval)
#  - Performance over sequence length (and more epochs)

n_test = 10000
n_train = 5000
n_val = 2000

vocab_size = 10000
max_len = 200
test_texts, test_labels = DataReader(0.5, dataset='test').take(n_test)

train_dr = DataReader(0.5)
train_texts, train_labels = train_dr.take(n_train)
val_texts, val_labels = train_dr.take(n_val)

tokenizer_train = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                        filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer_train.fit_on_texts(train_texts)


def get_dataset(texts, labels, tokenizer, batch_size=256, seq_len=max_len):
    tokens = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='post', maxlen=seq_len)
    dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
    batches = (dataset.shuffle(1000).padded_batch(batch_size, padded_shapes=([None], [])))
    return batches


train_batches = get_dataset(train_texts, train_labels, tokenizer_train)
validation_batches = get_dataset(val_texts, val_labels, tokenizer_train)
test_batches = get_dataset(test_texts, test_labels, tokenizer_train)


model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, 16),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(1)])

model.summary()

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='tensorboard_logs/1', histogram_freq=1)

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              )

history = model.fit(train_batches,
                    validation_data=validation_batches,
                    epochs=100,
                    callbacks=[tensorboard_callback])

results = model.evaluate(test_batches, verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
