"""
tf_multilayer_char_lstm.py

By applying function composition, implements a multilayer character LSTM with a multilayer 
classifier as a template for RNN construction.

Higher order functions simplify code while providing research flexibility for modifications, 
such as rewiring cells into new variants and trying different dropout schemes.

Python 3.5.2
TensorFlow 1.1.0
Includes a few code snippets from CS 20SI at http://web.stanford.edu/class/cs20si/ and udacity 
repository at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity.
"""

import math
import functools
import numpy as np
import tensorflow as tf

class CHAR_LSTM(object):
    """Build the graph of a character LSTM model."""
    def __init__(self, batch_size, vocab_size, lstm_topology, 
                 num_unrollings, classifier_topology,  
                 start_learning_rate, decay_steps, decay_rate, clip_norm):
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_unrollings = num_unrollings
        self.lstm_topology = lstm_topology
        self.classifier_topology = classifier_topology
        self.start_learning_rate = start_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.clip_norm = clip_norm
        self.global_step = tf.Variable(0, dtype=tf.int32, 
                                       trainable=False, 
                                       name='global_step')
     
    def _create_placeholders(self):
        """Define placeholders."""
        with tf.name_scope('dropout_prob'):
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        with tf.name_scope('data'):
            self.train_data = list()
            for ix in range(self.num_unrollings + 1):
                self.train_data.append(
                    tf.placeholder(tf.float32, 
                                   shape=[self.batch_size, self.vocab_size],
                                   name=('train_data_%d' %ix)))
            self.train_inputs = self.train_data[: self.num_unrollings]
            self.train_labels = self.train_data[1 :]              
            self.val_input = tf.placeholder(tf.float32, 
                                            shape=[1, self.vocab_size],
                                            name=('val_data_%d' %0))
   
    def _create_lstm_cells(self):
        """Define LSTM cells."""
        with tf.device('/cpu:0'):
            with tf.name_scope('lstm'):
                with tf.name_scope('saved_vars'):
                    (self.saved_train_outputs,
                     self.saved_train_states,
                     self.saved_val_outputs,
                     self.saved_val_states) = saved_variables(topology=self.lstm_topology, 
                                                              batch_size=self.batch_size)         
                    self.reset_saved_val_vars = tf.group(
                        *[output.assign(tf.zeros(tf.shape(output))) for output in self.saved_val_outputs],
                        *[state.assign(tf.zeros(tf.shape(state))) for state in self.saved_val_states])
                with tf.name_scope('vars'):  
                    lstm = lstm_fn(topology=self.lstm_topology)
                with tf.name_scope('ops'):
                    (self.train_outputs, 
                     self.val_output, 
                     self.to_save_train_outputs, 
                     self.to_save_train_states, 
                     self.to_save_val_outputs, 
                     self.to_save_val_states) = lstm(self.train_inputs, 
                                                     self.val_input, 
                                                     self.saved_train_outputs,              
                                                     self.saved_train_states, 
                                                     self.saved_val_outputs, 
                                                     self.saved_val_states) 
          
    def _create_classifier(self):
        """Define a neural network classifier."""
        with tf.device('/cpu:0'):
            with tf.name_scope('classifier'):
                with tf.name_scope('vars'):  
                    classifier = classifier_fn(
                        topology=self.classifier_topology, 
                        keep_prob=self.keep_prob)
                with tf.name_scope('train_ops'): 
                    # save the states and outputs from the last unrolling 
                    with tf.control_dependencies(
                        [saved_output.assign(to_save_output) for saved_output, to_save_output in 
                         zip(self.saved_train_outputs, self.to_save_train_outputs)] + 
                        [saved_state.assign(to_save_state) for saved_state, to_save_state in 
                         zip(self.saved_train_states, self.to_save_train_states)]):
                        # concatenate train outputs
                        self.train_logits = classifier(tf.concat(axis=0, values=self.train_outputs))
                        self.train_prediction = tf.nn.softmax(self.train_logits)
                with tf.name_scope('val_ops'): 
                    with tf.control_dependencies(
                        [saved_output.assign(to_save_output) for saved_output, to_save_output in 
                         zip(self.saved_val_outputs, self.to_save_val_outputs)] + 
                        [saved_state.assign(to_save_state) for saved_state, to_save_state in 
                         zip(self.saved_val_states, self.to_save_val_states)]):
                        self.val_logits = classifier(self.val_output)
                        self.val_prediction = tf.nn.softmax(self.val_logits)

    def _create_loss(self):
        """Define a loss function."""
        with tf.device('/cpu:0'):
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.train_logits, labels=tf.concat(axis=0, values=self.train_labels), name='loss'))

    def _create_training(self):
        """Define training ops."""
        with tf.device('/cpu:0'):
            self.learning_rate = tf.train.exponential_decay(
                learning_rate=self.start_learning_rate, 
                global_step=self.global_step, 
                decay_steps=self.decay_steps, 
                decay_rate=self.decay_rate,
                staircase=True, name='learning_rate')
            with tf.name_scope('gradient_descent'):
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                gradients, v = zip(* self.optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
                self.optimizer = self.optimizer.apply_gradients(
                    zip(gradients, v), global_step=self.global_step)
            
    def _create_summaries(self):
        """Define summaries."""
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            #add more summaries here if needed
            self.summary = tf.summary.merge_all()

    def build_graph(self):
        """Build the graph."""
        self._create_placeholders()
        self._create_lstm_cells()
        self._create_classifier()
        self._create_loss()
        self._create_training()
        self._create_summaries()
        
# auxiliary functions building variables

def weights_variable(shape, mean, stddev):
    """Build a weights variable."""
    return tf.Variable(tf.truncated_normal(
        shape=shape, mean=mean, stddev=stddev), name='weights')
    
def biases_variable(shape, val):
    """Build a biases variable."""
    return tf.Variable(tf.fill(dims=shape, value=val), name='biases')

def saved_variable(shape, name):
    """Build a state or output saving variable."""
    return tf.Variable(tf.zeros(shape=shape, dtype=tf.float32),
                       trainable=False, name=name)

#auxiliary functions for building a multilayer LSTM function

def cell_variables(in_dim, out_dim):
    """Build cell gate or update variables."""
    weights_shape = [in_dim, out_dim]
    weights_mean =  -0.1
    weights_stddev = 0.1 
    biases_shape = [out_dim]
    biases_val = 0.0
    weights = weights_variable(weights_shape, weights_mean, weights_stddev)
    biases = biases_variable(biases_shape, biases_val)
    return weights, biases

def saved_variables(topology, batch_size):
    """Build variables for saving previous output and state."""
    saved_train_outputs = list()
    saved_train_states = list()
    saved_val_outputs = list()
    saved_val_states = list()
    if len(topology) < 2:
        raise ValueError("Incompatible LSTM topology length.")
    for ix, num_nodes in enumerate(topology[1 :]):
        with tf.name_scope('train_l%d' %ix):
            saved_train_outputs.append(
                saved_variable(shape=[batch_size, num_nodes], name='output'))
            saved_train_states.append(
                saved_variable(shape=[batch_size, num_nodes], name='state'))
        with tf.name_scope('val_l%d' %ix):
            saved_val_outputs.append(
                saved_variable(shape=[1, num_nodes], name='output'))
            saved_val_states.append(
                saved_variable(shape=[1, num_nodes], name='state'))
    return saved_train_outputs, saved_train_states, saved_val_outputs, saved_val_states

def cell_fn(in_dim, out_dim):
    """Build a cell function."""
    with tf.name_scope('input_vars'):
        input_weights, input_biases = cell_variables(in_dim, out_dim)
    with tf.name_scope('forget_vars'):
        forget_weights, forget_biases = cell_variables(in_dim, out_dim)
    with tf.name_scope('output_vars'):
        output_weights, output_biases = cell_variables(in_dim, out_dim)
    with tf.name_scope('update_vars'):
        update_weights, update_biases = cell_variables(in_dim, out_dim)       
    def cell_ops(input_data, output_data, state):
        #concatenate last output and current input
        data = tf.concat(axis=1, values=[input_data, output_data])
        with tf.name_scope('input_ops'):
            input_gate = tf.sigmoid(tf.matmul(data, input_weights) + input_biases)
        with tf.name_scope('forget_ops'):
            forget_gate = tf.sigmoid(tf.matmul(data, forget_weights) + forget_biases)
        with tf.name_scope('output_ops'):
            output_gate = tf.sigmoid(tf.matmul(data, output_weights) + output_biases)
        with tf.name_scope('update_ops'):
            update = tf.matmul(data, update_weights) + update_biases
        with tf.name_scope('state_ops'):
            state = forget_gate * state + input_gate * tf.tanh(update)
            cell_output = output_gate * tf.tanh(state)
        return cell_output, state
    return (lambda input_data, output_data, state: 
            cell_ops(input_data, output_data, state))

def lstm_layer_fn(in_dim, out_dim, layer_name):
    """Build a function for an LSTM layer."""
    with tf.name_scope(layer_name + '_vars'):
        cell = cell_fn(in_dim=in_dim + out_dim, out_dim=out_dim)  
    def lstm_layer_ops(train_inputs, 
                       val_input, 
                       saved_train_outputs, 
                       saved_train_states, 
                       saved_val_outputs, 
                       saved_val_states):
        train_output = saved_train_outputs[0]
        train_state = saved_train_states[0]
        train_outputs = list()
        with tf.name_scope(layer_name + '_ops'):
            for ix, train_input in enumerate(train_inputs):
                with tf.name_scope('train_%d' %ix):
                    train_output, train_state = cell(input_data=train_input, 
                                                     output_data=train_output,
                                                     state=train_state)  
                    train_outputs.append(train_output)
            with tf.name_scope('val'):
                val_output, val_state = cell(input_data=val_input,
                                             output_data=saved_val_outputs[0], 
                                             state=saved_val_states[0])
        return (train_outputs, 
                val_output, 
                saved_train_outputs[1 :] + [train_output], 
                saved_train_states[1 :] + [train_state], 
                saved_val_outputs[1 :] + [val_output], 
                saved_val_states[1 :] + [val_state]) 
    return (lambda train_inputs, 
                   val_input, 
                   saved_train_outputs, 
                   saved_train_states, 
                   saved_val_outputs, 
                   saved_val_states: lstm_layer_ops(train_inputs, 
                                                    val_input, 
                                                    saved_train_outputs, 
                                                    saved_train_states, 
                                                    saved_val_outputs, 
                                                    saved_val_states))

def lstm_fn(topology):
    """Build a function for a multilayer LSTM by function composition."""
    fn_list = list()
    if len(topology) < 2:
        raise ValueError("Incompatible LSTM topology length.")
    def compose_fn(f, g):
        return (lambda x_1, x_2, x_3, x_4, x_5, x_6: 
                g(*f(x_1, x_2, x_3, x_4, x_5, x_6))) 
    for ix, (in_dim, out_dim) in enumerate(zip(topology[: len(topology) -1],
                                               topology[1 :])):
        layer_name = 'lstm_l%d' %ix
        fn_list.append(lstm_layer_fn(in_dim=in_dim, 
                                     out_dim=out_dim, 
                                     layer_name=layer_name))
    return functools.reduce(compose_fn, fn_list)

#auxiliary functions for building a classifier function

def classifier_variables(in_dim, out_dim):
    """Build neural network classifier variables."""
    weights_shape = [in_dim, out_dim]
    weights_mean = 0.0
    weights_stddev = 0.1 #1.0 / math.sqrt(float(in_dim)) #Xavier initialization
    biases_shape = [out_dim]
    biases_val = 0.1 #avoid dead neurons
    weights = weights_variable(weights_shape, weights_mean, weights_stddev)
    biases = biases_variable(biases_shape, biases_val)
    return weights, biases

def classifier_layer_fn(in_dim, out_dim, layer_name, fn=tf.nn.relu):
    """Build a function for a neural network classifier layer."""
    with tf.name_scope(layer_name + '_vars'):
        weights, biases = classifier_variables(in_dim, out_dim)   
    def nn_layer_ops(x):
        with tf.name_scope(layer_name + '_ops'):
            return fn(tf.matmul(x, weights) + biases)
    return lambda input_dataset: nn_layer_ops(input_dataset)

def classifier_fn(topology, keep_prob):
    """Build a function for a neural network classifier by function composition."""
    fn_list = list()
    if len(topology) < 2:
        raise ValueError("Incompatible topology length.") 
    def drop(x):
        return tf.nn.dropout(tf.nn.relu(x), keep_prob=keep_prob)
    def compose_fn(f, g):
        return lambda x: g(f(x)) 
    for c, (in_dim, out_dim) in enumerate(zip(topology[: len(topology) -1], 
                                              topology[1 :])):
        if c == (len(topology) - 2):
            layer_name = 'out'
            fn_list.append(classifier_layer_fn(in_dim, out_dim, 
                                               layer_name, fn=tf.identity))
        else:
            layer_name = 'hl%d' %c
            fn_list.append(classifier_layer_fn(in_dim, out_dim, 
                                               layer_name, fn=drop))
    return functools.reduce(compose_fn, fn_list)