from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from data_reader import DataReader
from preprocessing import simple_preprocess, simple_tokenize, preprocess
from baseline_tfidf.tfidf_svm import tfidf_svm

n_test = 10000

test_text, test_labels = DataReader(0.5, dataset='test').take(n_test)

s_test = set(test_text)
test_dataset = [simple_preprocess(t) for t in tqdm(test_text)]

accuracy_per_step_svm = []
steps = range(500, 10000, 500)
for n_obs in steps:
    texts, labels = DataReader(0.5).take(n_obs)

    s_train = set(texts)
    print(f"train/test overlaps: {len(s_train & s_test)}")  # todo: ensure 0 overlaps

    dataset = [simple_preprocess(t) for t in texts]

    svm_model = tfidf_svm(dataset, labels)

    predicted = svm_model.predict(test_dataset)
    acc = accuracy_score(test_labels, predicted)

    print(f"accuracy with {n_obs} training examples. svm: {acc}")
    accuracy_per_step_svm.append(acc)

df_plot = pd.DataFrame(data={'svm': accuracy_per_step_svm, 'training_examples': list(steps)})

sns.set()
sns_plot = sns.relplot(data=pd.melt(df_plot, ['training_examples']), x='training_examples',
                       y='value', hue='variable', kind="line")
sns_plot.savefig("baseline_per_examples.png")
