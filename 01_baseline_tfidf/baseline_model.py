from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

from data_reader import DataReader
from preprocessing import simple_preprocess, simple_tokenize

n_test = 1000

test_text, test_labels = DataReader(0.5).take(n_test)

accuracy_per_step = []
steps = range(50, 1500, 50)
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

    predicted_labels = []
    for pos_val, neg_val in zip(tfidf_pos[corpus_test], tfidf_neg[corpus_test]):
        pos_tot = sum([e[1] for e in pos_val])
        neg_tot = sum([e[1] for e in neg_val])
        predicted_labels.append(1 if (pos_tot - neg_tot) > 0 else 0)

    acc = accuracy_score(test_labels, predicted_labels)
    print(f"accuracy with {n_obs} training examples:  {acc}")
    accuracy_per_step.append(acc)

df_plot = pd.DataFrame(data={'accuracy': accuracy_per_step, 'training_examples': list(steps)})

sns.set()
sns_plot = sns.relplot(data=df_plot, x='training_examples', y='accuracy', kind="line",)
sns_plot.savefig("baseline_per_examples.png")
