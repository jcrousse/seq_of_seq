import os
import warnings
import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_interface import DataReader
from config.config import PKL_LOCATION, TXT_LOCATION, EXP_DIRS


def _check_sub_dir(path):
    if all([os.path.isdir(os.path.join(path, d)) for d in EXP_DIRS]):
        return True
    else:
        warnings.warn(f"{path} exists but does not contain {EXP_DIRS} subfolders")
        return False


def _check_datasert_dir(path):
    return True if os.path.isdir(path) and _check_sub_dir(path) else False


def _prep_txt_dataset(dataset_name, n_train, n_val, n_test):
    test_texts, test_labels = DataReader(0.5, dataset=dataset_name, subset='test').take(n_test)
    train_dr = DataReader(0.5, dataset=dataset_name)
    train_texts, train_labels = train_dr.take(n_train + n_val)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=n_val)
    return [(train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)]


def generator_factory(files_to_read):
    def gn():
        for file in files_to_read:
            with open(file, 'rb') as rf:
                obs_data = pickle.load(rf)
                yield {'input': obs_data['input'], 'output': obs_data['output']}
    return gn


def get_filpaths_pkl(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(f)[1] == '.pkl']


def filelist_to_dataset(filelist):
    return tf.data.Dataset.from_generator(generator_factory(filelist), ({'input': tf.float32},
                                                                        {'output': tf.int32}))


def _prep_tf_dataset(dataset_name, n_train, n_val, n_test):
    dataset_path = os.path.join(PKL_LOCATION, dataset_name)
    all_train_files = get_filpaths_pkl(os.path.join(dataset_path, 'train'))[:n_train + n_val]
    test_files = get_filpaths_pkl(os.path.join(dataset_path, 'test'))[:n_test]
    train_files, val_files = train_test_split(all_train_files, test_size=n_val)
    return [filelist_to_dataset(fl) for fl in [train_files, val_files, test_files]]


def get_dataset(dataset_name, n_train, n_val, n_test, pre_calc_embedd=False):
    """Checks if pre-processed (embeddings) dataset exists as pickle and loads as tf datasets,
    if not, check if files raw exists as txt and loads as in-memory lists"""
    base_path = TXT_LOCATION
    load_func = _prep_txt_dataset
    if pre_calc_embedd:
        base_path = PKL_LOCATION
        load_func = _prep_tf_dataset

    path = os.path.join(base_path, dataset_name)
    if _check_datasert_dir(path):
        return load_func(dataset_name, n_train, n_val, n_test)
