import re
import unidecode
import spacy
import tensorflow as tf
from spacy.pipeline import Sentencizer

from config.config import PARAGRAPH_SEPARATOR

sentencizer = Sentencizer()
nlp = spacy.load('en_core_web_sm')
sentence_splitter = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
sentence_splitter.add_pipe(sentencizer)


def simple_preprocess(text):
    """remove special characters and puts everything in lowercase
    """
    return re.sub(r'[^a-zA-Z0-9_\s]', '', unidecode.unidecode(text)).lower()


def simple_tokenize(text):
    """
    whitespace tokenization
    """
    return text.split(' ')


def preprocess(text):
    doc = nlp(simple_preprocess(text))
    processed_tokens = [t.lemma_ for t in doc if not t.is_stop]
    return " ".join(processed_tokens)


def split_all_sentences(text):
    """
    split by sentences, then by word. Returns text of fixed length with padded sentences.
    :param text: string to split
    :return:
    """
    sentences = [str(sent) for sent in sentence_splitter(text).sents]
    return sentences


def split_paragraphs(text: str):
    return text.split(PARAGRAPH_SEPARATOR)


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = f.read()
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    return tokenizer


def load_or_fit_tokenizer(tokenizer_dir, vocab_size, corpus=None, **_):
    tokenizer_path = tokenizer_dir / 'tokenizer.json'
    if tokenizer_path.exists():
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        assert corpus is not None, "corpus must be provided if not tokenizer.json file in tokenizer_dir"
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                          filters=r'><!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        tokenizer.fit_on_texts(corpus)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        with open(tokenizer_path, 'w') as f:
            f.write(tokenizer.to_json())
    return tokenizer


def get_padded_sequences(texts, tokenizer, seq_len=200, split_sentences=False, sent_len=20,
                         sent_splitter=split_all_sentences, **_):
    if split_sentences:
        sentences = [sent_splitter(t) for t in texts]

        def sent_padding(s):
            return tf.keras.preprocessing.sequence.pad_sequences(s, padding='post', maxlen=sent_len)
    else:
        sentences = [[t] for t in texts]

        def sent_padding(s):
            return s

    tokenized_sentences = [tokenizer.texts_to_sequences(text) for text in sentences]
    padded_sentences = [sent_padding(tokens) for tokens in tokenized_sentences]
    preprocessed_texts = [[w for sentences in sent_list for w in sentences] for sent_list in padded_sentences]
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(preprocessed_texts, padding='post', maxlen=seq_len)

    return padded_sequences, [tokenizer.sequences_to_texts([list(e) for e in ps]) for ps in padded_sentences]


def get_dataset(data, batch_size=256, concat_outputs=False, n_outputs=1, out_shape=1):

    out_s = [out_shape, ]
    if out_s == [1]:
        out_s = []
    if isinstance(data, tf.data.Dataset):
        dataset = data
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
    if concat_outputs:
        dataset = dataset.map(lambda x, y: (x,
                                            {'output': [y['output']] * n_outputs,
                                             'output_2':  [y['output_2']] * n_outputs}))
    batches = dataset.shuffle(1000).padded_batch(batch_size,
                                                 padded_shapes=({'input': [None]}, {'output': out_s,
                                                                                    'output_2': out_s}))
    return batches


if __name__ == '__main__':
    example_text = "this is a text! it has four sentences of different lengths. a question? Sure"
    print(split_all_sentences(example_text))
