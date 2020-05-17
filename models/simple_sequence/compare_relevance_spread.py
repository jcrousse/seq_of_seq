import numpy as np
import seaborn as sns
import pandas as pd
from itertools import combinations

from data_interface import DataReader
from models.simple_sequence.experiment import Experiment

MODELS = ['score200_20', 'l_concat200_20']
predict_text, predict_labels = DataReader(0.5, subset='test').take(1000)


def get_relevance(model_name, texts):
    model = Experiment(model_name)
    return model.predict_layer(texts, layer="score")


def get_distrib(mat):
    return np.sum(np.square(mat - np.ones(mat.shape) / mat.shape[1]), axis=1)


if __name__ == '__main__':
    m_to_distrib = {model: get_distrib(get_relevance(model, predict_text)) for model in MODELS}

    for model1, model2 in combinations(MODELS, 2):
        individual_diffs = m_to_distrib[model1] - m_to_distrib[model2]

        print(f"item with maximal difference between {model1} and {model2}:",
              np.argmax(individual_diffs))

        sns.set()
        plot = sns.scatterplot(m_to_distrib[model1], m_to_distrib[model2])
        plot.figure.savefig(f"{model1}_vs_{model2}.png")
        _ = 1

        line_df = pd.DataFrame(data={f'diff_{model1}_{model2}': sorted(individual_diffs)})
        plot_line = sns.relplot(data=line_df)
        plot_line.savefig(f"{model1}_vs_{model2}_line.png")




