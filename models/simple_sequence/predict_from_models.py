from data_reader import DataReader
from models.simple_sequence.experiment import Experiment

MODEL_LOAD = 'testus'


model = Experiment(MODEL_LOAD)
predict_text, predict_labels = DataReader(0.5, dataset='test').take(1)
predictions = model.predict(predict_text)
bin_pred = [0 if p < 0 else 1 for p in predictions]
relevance = model.predict_layer(predict_text, layer="score")
_ = 1
