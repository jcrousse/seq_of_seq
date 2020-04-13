import os
from random import shuffle
from config.config import TRAIN_SET, TEST_SET


class DataReader:
    def __init__(self, pos_pct=0.5, dataset='train'):
        """
        defines a data reader (generator) with a certain percentage of positive examples.
        apporach to reach the percentage is most basic:
        we assume pct is multiplied by N to make it a round number.
        This object will yield pos_pct * N positive example, and (1-pos_pct) * N negative ones.
        :param pos_pct: percentage of positive examples to return
        """

        decimal_n = len(str(pos_pct)) - str(pos_pct).find(".")
        total_ex = 10 ** (decimal_n - 1)
        self.total_pos = int(pos_pct * total_ex)
        self.total_neg = int((1 - pos_pct) * total_ex)

        self.current_idx = 0
        self.next_labels_vector = []
        self.update_next_labels()

        self.data_dirs = TRAIN_SET if dataset == 'train' else TEST_SET

        self.example_files = {
            label: iter([self.data_dirs[label] + '/' + str(e) for e in os.listdir(self.data_dirs[label])])
            for label in [0, 1]
        }

    def update_next_labels(self):
        next_vector = [0] * self.total_neg + [1] * self.total_pos
        shuffle(next_vector)
        self.next_labels_vector = next_vector

    def read_example(self, label):
        file = next(self.example_files[label])
        with open(file, 'r') as f:
            return f.read()

    def udpate_idx(self):
        idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.next_labels_vector)
        return idx

    def __call__(self):
        label = self.next_labels_vector[self.udpate_idx()]
        return self.read_example(label), label

    def take(self, n):
        dataset = []
        for i in range(int(n)):
            dataset.append(self())
        split = list(zip(*dataset))
        return list(split[0]), list(split[1])
