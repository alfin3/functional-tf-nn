"""
tf_data.py

Implements data pre-processing functions.

Python 3.5.2
TensorFlow 0.12.1
Includes a few code snippets from CS 20SI at http://web.stanford.edu/class/cs20si/.
"""

import os
import numpy as np
from six.moves import cPickle as pickle

DATA_ROOT = '/home/ubuntu/covertype'

TRAIN_DATA = 'attr_trainset_NN.csv'
TRAIN_LABELS = 'class_trainset_NN.csv'
VAL_DATA = 'attr_valset_NN.csv'
VAL_LABELS = 'class_valset_NN.csv'
TEST_DATA = 'attr_testset_NN.csv'
TEST_LABELS = 'class_testset_NN.csv'

TRAIN_SET_SIZE = 11340
VAL_SET_SIZE = 3780
TEST_SET_SIZE = 565892
NUM_ATTR = 54
NUM_CLASSES = 7

def parse_input(data_file, fn_type, num_examples, num_dim):
    """
    Parses the data file and creates a dataset of num_examples instances.
    data = parse_input(open('attr_testset_NN.csv', "r"), np.float32, 5000, 54)
    """
    dataset = np.ndarray(shape=(num_examples, num_dim), dtype=fn_type)
    ct = 0
    for line in data_file:
        instance = line.split(",")
        dataset[ct, :] = instance
        ct += 1
        if not num_examples is None and ct >= num_examples:
            break
    print('File: ', data_file.name)        
    print('Shape: ', dataset.shape)
    print('Mean: ', np.mean(dataset))
    print('Sdev: ', np.std(dataset))
    return dataset

def randomize(dataset, labels):
    """Randomizes the dataset and labels while preserving alignment."""
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def create_pickle(file_name):
    """Creates a pickle file from train, test, and validation sets.
    create_pickle('covertype.pickle')"""
    pickle_file = os.path.join(DATA_ROOT, file_name)
    try:
        f = open(pickle_file, 'wb')
        train_data = parse_input(open(TRAIN_DATA, "r"), 
                                np.float32, TRAIN_SET_SIZE, NUM_ATTR)
        val_data = parse_input(open(VAL_DATA, "r"), 
                                np.float32, VAL_SET_SIZE, NUM_ATTR)
        test_data = parse_input(open(TEST_DATA, "r"), 
                                np.float32, TEST_SET_SIZE, NUM_ATTR)
        train_labels = parse_input(open(TRAIN_LABELS, "r"), 
                                np.int16, TRAIN_SET_SIZE, 1)
        val_labels = parse_input(open(VAL_LABELS, "r"), 
                                np.int16, VAL_SET_SIZE, 1)
        test_labels = parse_input(open(TEST_LABELS, "r"), 
                                np.int16, TEST_SET_SIZE, 1)
        shuffled_train_data, shuffled_train_labels = randomize(train_data, train_labels)
        shuffled_val_data, shuffled_val_labels = randomize(val_data, val_labels)
        shuffled_test_data, shuffled_test_labels = randomize(test_data, test_labels)
        save = {
            'train_dataset': shuffled_train_data,
            'train_labels': (np.arange(NUM_CLASSES) == 
                             shuffled_train_labels).astype(np.float32),
            'val_dataset': shuffled_val_data,
            'val_labels': (np.arange(NUM_CLASSES) == 
                           shuffled_val_labels).astype(np.float32),
            'test_dataset': shuffled_test_data,
            'test_labels': (np.arange(NUM_CLASSES) == 
                            shuffled_test_labels).astype(np.float32),
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    statinfo = os.stat(pickle_file)
    print('Pickle size:', statinfo.st_size)
    
def load_pickle(pickle_file):
    """Loads a pickle file."""
    try:
        with open(pickle_file, 'rb') as f:
            pickle_object = pickle.load(f)         
    except Exception as e:
        print("Unable to load from", pickle_file, ":", e)
        raise
    return pickle_object

def load_data(pickle_file):
    """Loads a data from a pickle file."""
    print("Loading data...")
    dict_dataset = load_pickle(pickle_file)
    train_dataset = dict_dataset['train_dataset']
    val_dataset = dict_dataset['val_dataset']
    test_dataset = dict_dataset['test_dataset']
    train_labels = dict_dataset['train_labels']
    val_labels = dict_dataset['val_labels']
    test_labels = dict_dataset['test_labels']
    del dict_dataset
    print("Data loaded.")
    print("Training set: ", train_dataset.shape, train_labels.shape)
    print("Validation set: ", val_dataset.shape, val_labels.shape)
    print("Test set: ", test_dataset.shape, test_labels.shape)
    return (train_dataset, val_dataset, test_dataset, 
            train_labels, val_labels, test_labels)  
