from pathlib import Path
from shutil import rmtree

import numpy as np

from data_interface import DataReader
from models.simple_sequence.experiment import Experiment
from utils.write_to_html import write_to_html

MODELS_LOAD = ['l_concat_200_20']
OVERWRITE = True
NUM_PRED = 30

DATASET = "aclImdb"  # "P4_from1200_vocab200_fromPNone_noextra"

if __name__ == '__main__':

    predict_text, predict_labels = DataReader(0.5, dataset=DATASET, subset='test').take(NUM_PRED)

    for m_name in MODELS_LOAD:
        model = Experiment(m_name)
        path = Path('html_out') / m_name
        if not path.exists() or OVERWRITE:
            if path.exists():
                rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        prediction, sentences = model.predict(predict_text, return_sentences=True)
        relevance = model.predict_layer(predict_text, layer="score")

        sum_logits = np.sum(prediction, axis=1)
        logits = prediction[:, 0]
        probs = [1 / (1 + np.exp(-e)) for e in logits]
        diff_sq_round = [round((v1 - v2) ** 2, 2) for v1, v2 in zip(probs, predict_labels)]
        bin_pred = [1 if e > 0 else 0 for e in logits]

        for idx, scores, sents, diff, pred, actual in zip(range(len(sentences)), relevance, sentences, diff_sq_round, bin_pred,
                                                          predict_labels):
            write_to_html(sents, scores, f"{diff}_ex{idx}_act{actual}_pred{pred}.html", out_dir=path)
            # print(f"relevance: {score}, sentence: {sent}")
