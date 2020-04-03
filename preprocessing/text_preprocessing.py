import re
import unidecode


def simple_preprocess(text):
    """remove special characters and puts everything in lowercase
    """
    return re.sub(r'[^a-zA-Z0-9_\s]', '', unidecode.unidecode(text)).lower()


def simple_tokenize(text):
    """
    whitespace tokenization
    """
    return text.split(' ')