import tensorflow as tf

from data_reader import DataReader


# todo now: tensorboard output because it is nicer. Clean up this crap.
n_test = 10000
n_train = 5000
n_val = 2000

vocab_size = 10000
max_len = 200
test_texts, test_labels = DataReader(0.5, dataset='test').take(n_test)

train_dr = DataReader(0.5)
train_texts, labels = train_dr.take(n_train)
val_texts, val_labels = train_dr.take(n_val)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_texts)

train_tokens = tokenizer.texts_to_sequences(train_texts)
test_tokens = tokenizer.texts_to_sequences(test_texts)
val_tokens = tokenizer.texts_to_sequences(val_texts)

pad_seq_train = tf.keras.preprocessing.sequence.pad_sequences(train_tokens, padding='post', maxlen=max_len)
pad_seq_val = tf.keras.preprocessing.sequence.pad_sequences(val_tokens, padding='post', maxlen=max_len)
pad_seq_test = tf.keras.preprocessing.sequence.pad_sequences(test_tokens, padding='post', maxlen=max_len)

print("max length:", max(len(t) for t in pad_seq_train))

dataset_train = tf.data.Dataset.from_tensor_slices((pad_seq_train, labels))
dataset_val = tf.data.Dataset.from_tensor_slices((pad_seq_val, val_labels))
dataset_test = tf.data.Dataset.from_tensor_slices((pad_seq_test, test_labels))
train_batches = (
    dataset_train
    .shuffle(1000)
    .padded_batch(256, padded_shapes=([None], [])))

validation_batches = (
    dataset_val
    .shuffle(1000)
    .padded_batch(256, padded_shapes=([None], [])))

test_batches = (
    dataset_test
    .shuffle(1000)
    .padded_batch(256, padded_shapes=([None], [])))

for example_batch, label_batch in dataset_train.take(2):
    print(example_batch)
    print(label_batch)
    print("Batch shape:", example_batch.shape)
    print("label shape:", label_batch.shape)

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, 16),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(1)])

model.summary()

model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_batches,
                    validation_data=validation_batches,
                    epochs=100)

results = model.evaluate(test_batches, verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))