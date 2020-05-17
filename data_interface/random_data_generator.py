import os
from pathlib import Path
from collections import Counter
from shutil import rmtree

from tqdm import tqdm
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

from config.config import DIR_LABEL_MAP, BASE_PATH, PARAGRAPH_SEPARATOR
from data_interface import DataReader
from preprocessing.text_preprocessing import split_all_sentences

N_EXAMPLES = 20000
N_PARAGRAPHS = 4
SUBSET = 'test'

N_SOURCE_EXAMPLES = 1200
VOCAB_SIZE = 200
OVERWRITE = True

EXTRA_SOURCES = None #['data/kafka_trial.txt']
MAX_RANDOM_PARAGRAPHS = None
REPLACE = False

if MAX_RANDOM_PARAGRAPHS is None:
    len_random_p = N_EXAMPLES * (N_PARAGRAPHS - 1)
else:
    len_random_p = MAX_RANDOM_PARAGRAPHS

extra = 'noextra'
if EXTRA_SOURCES is not None:
    extra = 'extra'
LABEL_DIR_MAP = {v: k for k, v in DIR_LABEL_MAP.items()}


def prep_directories():
    dataset_name = f"P{N_PARAGRAPHS}_from{N_SOURCE_EXAMPLES}_vocab{VOCAB_SIZE}_fromP{MAX_RANDOM_PARAGRAPHS}_{extra}"
    target_path = Path(BASE_PATH) / dataset_name / SUBSET
    if target_path.exists():
        if OVERWRITE:
            rmtree(target_path)
        else:
            raise(FileExistsError(f"Dataset {target_path} already exists"))

    target_path.mkdir(parents=True)
    label_dir_map = {}
    for subdir, lab in DIR_LABEL_MAP.items():
        path = target_path / subdir
        path.mkdir()
        label_dir_map[lab] = path
    return label_dir_map


def counter_to_probs(cnt):
    return np.array(list(cnt.values())) / sum(list(cnt.values()))


def random_pick(cnt, n):
    probs = counter_to_probs(cnt)
    return list(np.random.choice(list(cnt.keys()), n, p=probs))


def generate_n_paragraphs(n, texts):
    sentences = [split_all_sentences(t) for t in texts]
    tokenized_sentences = [tokenizer.texts_to_sequences(s) for s in sentences]
    paragraph_len_count = Counter([len(p) for p in sentences])
    sentence_len_count = Counter([len(s) + 1 for p in tokenized_sentences for s in p])
    word_count = Counter([w for p in tokenized_sentences for s in p for w in s])

    sent_per_paragraph = random_pick(paragraph_len_count, n)
    w_per_sent = [random_pick(sentence_len_count, n) for n in sent_per_paragraph]
    words = [[random_pick(word_count, n) for n in sent_len] for sent_len in w_per_sent]

    paragraphs_as_text = [". ".join(tokenizer.sequences_to_texts(sents)) for sents in words]

    return paragraphs_as_text


if __name__ == '__main__':
    label_to_dir = prep_directories()
    texts_tk, _ = DataReader(0.5, subset=SUBSET).take(N_SOURCE_EXAMPLES)

    if EXTRA_SOURCES is not None:
        for extra_sc in EXTRA_SOURCES:
            with open(extra_sc, 'rb') as f:
                texts_tk.append(f.read().decode())
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters=r'><!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(texts_tk)
    reviews, labels = DataReader(0.5, subset=SUBSET).take(N_EXAMPLES)

    random_paragraphs = generate_n_paragraphs(len_random_p, reviews)

    total_random_p_len = N_EXAMPLES * (N_PARAGRAPHS - 1)
    all_random_paragraphs = list(np.random.choice(random_paragraphs, total_random_p_len, replace=REPLACE))

    paragraphs_idx = list(range(0, total_random_p_len, N_PARAGRAPHS - 1))
    for idx, text, label, filename in tqdm(zip(paragraphs_idx, reviews, labels, range(len(reviews))), total=N_EXAMPLES):
        idx_real_p = np.random.randint(N_PARAGRAPHS + 1)
        all_paragraphs = all_random_paragraphs[idx:idx+N_PARAGRAPHS - 1]
        all_paragraphs.insert(idx_real_p, text)
        full_text = PARAGRAPH_SEPARATOR.join(all_paragraphs)

        with open(os.path.join(label_to_dir[label], str(filename) + '.txt'), 'w') as f:
            f.write(full_text)
