from data_reader import DataReader
from models.simple_sequence.experiment import Experiment

MODEL_LOAD = 'testus'


model = Experiment(MODEL_LOAD)
predict_text, predict_labels = DataReader(0.5, dataset='test').take(100)
print(model.predict(predict_text))
