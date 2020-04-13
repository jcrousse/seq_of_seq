import re
import unidecode
import spacy
from spacy.pipeline import Sentencizer


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


def split_sentences(text):
    """
    split by sentences, then by word. Returns text of fixed length with padded sentences.
    :param text: string to split
    :param split: charachters marking the end of a sentence
    :return:
    """
    sentences = [str(sent) for sent in sentence_splitter(text).sents]
    return sentences


if __name__ == '__main__':
    example_text = "this is a text! it has four sentences of different lengths. a question? Sure"
    print(split_sentences(example_text))
