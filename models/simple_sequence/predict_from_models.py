from data_reader import DataReader
from models.simple_sequence.experiment import Experiment
from utils.write_to_html import write_to_html

MODEL_LOAD = 'testus'


model = Experiment(MODEL_LOAD)
predict_text, predict_labels = DataReader(0.5, dataset='test').take(100)
_, sentences = model.predict(predict_text, return_sentences=True)

relevance = model.predict_layer(predict_text, layer="score")

idx = 0
for scores, sents in zip(relevance, sentences):
    write_to_html(sents, scores, f"{idx}.html")
    idx = idx+1
    # print(f"relevance: {score}, sentence: {sent}")
_ = 1
