from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

from data_reader import DataReader
from preprocessing import simple_preprocess, simple_tokenize
from baseline_tfidf.tfidf_svm import tfidf_svm

n_test = 10000

test_text, test_labels = DataReader(0.5).take(n_test)

accuracy_per_step = []
accuracy_per_step_svm = []
steps = range(50, 2000, 50)
for n_obs in steps:
    texts, labels = DataReader(0.5).take(n_obs)

    dataset = [simple_tokenize(simple_preprocess(t)) for t in texts]
    test_dataset = [simple_tokenize(simple_preprocess(t)) for t in test_text]

    dct = Dictionary(dataset)
    corpus = [dct.doc2bow(document) for document in dataset]
    corpus_test = [dct.doc2bow(document) for document in test_dataset]

    #todo: use numpy arrays for corpus. More efficent
    corpus_pos = [doc for idx, doc in enumerate(corpus) if labels[idx] == 1]
    corpus_neg = [doc for idx, doc in enumerate(corpus) if labels[idx] == 0]

    tfidf_pos = TfidfModel(corpus_pos)
    tfidf_neg = TfidfModel(corpus_neg)

    sklearn_model = tfidf_svm(texts, labels)

    predicted_labels = []
    for pos_val, neg_val in zip(tfidf_pos[corpus_test], tfidf_neg[corpus_test]):
        pos_tot = sum([e[1] for e in pos_val])
        neg_tot = sum([e[1] for e in neg_val])
        predicted_labels.append(1 if (pos_tot - neg_tot) > 0 else 0)

    predicted_sklearn = sklearn_model.predict(test_text)
    acc_sklearn = accuracy_score(test_labels, predicted_sklearn)

    acc = accuracy_score(test_labels, predicted_labels)
    print(f"accuracy with {n_obs} training examples. base:  {acc}, sklearn_svm: {acc_sklearn}")
    accuracy_per_step.append(acc)
    accuracy_per_step_svm.append(acc_sklearn)

df_plot = pd.DataFrame(data={'base_model': accuracy_per_step, 'sk_svm': accuracy_per_step_svm, 'training_examples': list(steps)})

sns.set()
sns_plot = sns.relplot(data=pd.melt(df_plot, ['training_examples']), x='training_examples',
                       y='value', hue='variable', kind="line")
sns_plot.savefig("baseline_per_examples.png")
