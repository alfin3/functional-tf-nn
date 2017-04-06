"""
tf_nn.py

Implements fully connected neural networks.

Python 3.5.2
TensorFlow 0.12.1
Includes a few code snippets from CS 20SI at http://web.stanford.edu/class/cs20si/.
"""

import math
import functools
import numpy as np
import tensorflow as tf

class NNModel(object):
    """Builds the graph for a NN model."""
    def __init__(self, topology_list, model_name, start_learning_rate,
                decay_rate, decay_steps):
        self.topology_list = topology_list
        self.model_name = model_name 
        self.start_learning_rate = start_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.global_step = tf.Variable(0, dtype=tf.int32, 
                                       trainable=False, name='global_step')

    def _create_placeholders(self):
        """Define placeholders."""
        with tf.name_scope('placeholders'):
            num_attr = self.topology_list[0]
            num_classes = self.topology_list[-1]
            self.dataset_placeholder = tf.placeholder(
                tf.float32, shape=(None, num_attr), name='dataset')
            self.labels_placeholder = tf.placeholder(
                tf.float32, shape=(None, num_classes), name='labels')
            self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob')

    def _create_model(self):
        """Define a model."""
        with tf.device('/cpu:0'):
            with tf.name_scope(self.model_name):
                model = model_fn(topology_lst=self.topology_list, 
                                 keep_prob=self.keep_prob_placeholder)
                self.logits = model(self.dataset_placeholder)      
     
    def _create_loss(self):
        """Define a loss function. """
        with tf.device('/cpu:0'):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.labels_placeholder, 
                        logits=self.logits, name='loss'))
                
    def _create_training_op(self):
        """Define the training op."""
        with tf.device('/cpu:0'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.start_learning_rate, 
                global_step=self.global_step, 
                decay_steps=self.decay_steps, 
                decay_rate=self.decay_rate,
                staircase = True, name='learning_rate')
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(loss=self.loss, 
                                                  global_step=self.global_step)
            
    def _create_evaluation(self):
        """Define evaluation functions."""
        with tf.name_scope('predictions'):
            predictions = tf.nn.softmax(logits=self.logits, name='predictions')
            correct_predictions = tf.equal(tf.argmax(self.labels_placeholder, 1),
                                           tf.argmax(predictions, 1), 
                                           name='correct_predictions')
            self.accuracy = tf.multiply(
                100.0, tf.reduce_mean(tf.cast(correct_predictions, 
                                              tf.float32), name='accuracy'))
            self.accuracy_count = tf.reduce_sum(
                tf.cast(correct_predictions, tf.int32), name='accuracy_count')

    def _create_summaries(self):
        """Define summaries."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """Build the graph."""
        self._create_placeholders()
        self._create_model()
        self._create_loss()
        self._create_training_op()
        self._create_evaluation()
        self._create_summaries()        
        
# NN model functions

def weights_variable(shape, stddev):
    """Build a weights variable."""
    return tf.Variable(tf.truncated_normal(
        shape=shape, stddev=stddev), name='weights')
    
def biases_variable(shape, val):
    """Build a biases variable."""
    return tf.Variable(tf.fill(dims=shape, value=val), name='biases')

def nn_layer_fn(in_dim, out_dim, layer_name, fn=tf.nn.relu):
    """Builds a function of an nn layer, fully connected to the previous layer."""
    with tf.name_scope(layer_name + '_vars'):
        weights_shape = [in_dim, out_dim]
        weights_stddev = 1.0 / math.sqrt(float(in_dim)) #Xavier initialization
        biases_shape = [out_dim]
        biases_val = 0.1 #to avoid dead neurons
        weights = weights_variable(weights_shape, weights_stddev)
        biases = biases_variable(biases_shape, biases_val)
    def nn_layer_ops(x):
        with tf.name_scope(layer_name + '_ops'):
            return fn(tf.matmul(x, weights) + biases)
    return lambda input_dataset: nn_layer_ops(input_dataset)

def model_fn(topology_lst, keep_prob):
    """Builds a function of an nn model."""
    fn_list = list()
    if len(topology_lst) < 2:
        raise ValueError("Incompatible topology length.") 
    def drop(x):
        return tf.nn.dropout(tf.nn.relu(x), keep_prob=keep_prob)
    def compose_fn(f, g):
        return lambda x: g(f(x)) 
    for c, (in_dim, out_dim) in enumerate(zip(topology_lst[: len(topology_lst) -1], 
                                                topology_lst[1 :])):
        if c == (len(topology_lst) - 2):
            layer_name = 'out'
            fn_list.append(nn_layer_fn(in_dim, out_dim, layer_name, fn=tf.identity))
        else:
            layer_name = 'hl%d' %c
            fn_list.append(nn_layer_fn(in_dim, out_dim, layer_name, fn=drop))
    return functools.reduce(compose_fn, fn_list)