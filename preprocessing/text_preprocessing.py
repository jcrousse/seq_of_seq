import re
import unidecode
import spacy

nlp = spacy.load('en_core_web_sm')


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