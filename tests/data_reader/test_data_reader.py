from unittest import TestCase
from data_reader import DataReader


class TestDataReader(TestCase):
    def test_init_val(self):
        dr = DataReader(0.999)
        self.assertEqual(sum(dr.next_labels_vector), 999)

    def test_read_files(self):
        dr = DataReader(0.5)
        examples = []
        for i in range(20):
            examples.append(dr())
        self.assertEqual(len(examples), 20)

    def test_t_set(self):
        dr = DataReader(subset='test')
        examples = []
        for i in range(20):
            examples.append(dr())
        self.assertEqual(len(examples), 20)
