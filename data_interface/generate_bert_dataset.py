import os

import tensorflow as tf
import numpy as np
import spacy
import pickle
from data_interface import DataReader
from preprocessing.text_preprocessing import split_paragraphs

nlp = spacy.load('en_trf_bertbaseuncased_lg')
SOURCE_DATASET = "P4_from1200_vocab200_fromPNone_noextra"
DATA_READER = DataReader(dataset=SOURCE_DATASET)
DEST_PATH = "data/bert"
MAX_OBS = 100

# todo: create train, val, test datasets + read them for experiment training


def sentence_embedding(text, text_splitter=split_paragraphs, sentence_len=200, embed_len=768):
    texts_splitted = text_splitter(text)
    berted_sentences = [nlp(s).tensor for s in texts_splitted]
    return tf.keras.preprocessing.sequence.pad_sequences(berted_sentences, maxlen=sentence_len, dtype="float32",
                                                         value=np.zeros(embed_len), padding="post")


def texts_embeddings(text, text_splitter=split_paragraphs, seq_len=4, sentence_len=200, embed_len=768):
    padded_sentences = sentence_embedding(text, text_splitter=text_splitter, sentence_len=sentence_len,
                                          embed_len=embed_len)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences([padded_sentences], maxlen=seq_len, dtype="float32",
                                                                value=np.zeros((sentence_len, embed_len)),
                                                                padding='post')[0]
    return padded_text


def obs_generator():
    while True:
        text, label = DATA_READER()
        yield {'input': texts_embeddings(text), 'output': label, 'output_2': label}


def get_pickle_dataset_generator(path):
    all_files = os.listdir(path)

    def ze_generator():
        scope_files = [fl for fl in all_files if os.path.splitext(fl)[1] == '.pkl']
        for file in scope_files:
            with open(os.path.join(path, file), 'rb') as rf:
                yield pickle.load(rf)
    return ze_generator


if __name__ == '__main__':
    dataset_dir = os.path.join(DEST_PATH, SOURCE_DATASET)
    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    for i, bert_encoding in enumerate(obs_generator()):
        if i > MAX_OBS:
            break
        with open(os.path.join(dataset_dir,  f'obs{i}.pkl'), 'wb') as f:
            pickle.dump(bert_encoding, f)
    tf_dataset = tf.data.Dataset.from_generator(get_pickle_dataset_generator(dataset_dir), {'input': tf.float32,
                                                                                            'output': tf.int32,
                                                                                            'output_2': tf.int32})
    for item in tf_dataset:
        print(item)
