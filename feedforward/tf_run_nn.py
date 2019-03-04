#! /usr/bin/env python

"""
tf_run_nn.py

Implements model training and evaluation functions.

Python 3.5.2
TensorFlow 0.12.1
Includes a few code snippets from CS 20SI at http://web.stanford.edu/class/cs20si/.
"""
import os
import numpy as np
import tensorflow as tf
import tf_nn 
from tf_data import load_data

#data
DATA_ROOT = '/home/ubuntu/covertype'
PICKLE_DATA_FILE = os.path.join(DATA_ROOT, 'covertype.pickle')

#nn topology
#NN_TOPOLOGY_LST format: [num_inputs, size_hl1,..., size_hlX, num_labels]
NN_TOPOLOGY_LST = [54, 
                   1024, 1024, 1024, 1024,  
                   512, 256, 128, 64,
                   7]

#model name
NN_MODEL_NAME = 'nn_model_hl8_f'

#records
RECORDS_ROOT = '/home/ubuntu/covertype/'
CHECKPOINT_FOLDER = os.path.join(RECORDS_ROOT, 'checkpoints', NN_MODEL_NAME, '')
GRAPH_FOLDER = os.path.join(RECORDS_ROOT, 'graphs', NN_MODEL_NAME, '')

#stochastic sampling
BATCH_SIZE = 128
NUM_STEPS = 10001

#learning rate
START_LEARNING_RATE = 0.5
DECAY_RATE = 0.75
DECAY_STEPS = 5000

#dropout regularization
TRAIN_KEEP_PROB = 0.5
EVAL_KEEP_PROB = 1.0

def train(model, pickle_data_file, num_steps, checkpoint_folder, graph_folder):
    """The main training function."""
    saver = tf.train.Saver() 
    (train_dataset, val_dataset, test_dataset, 
     train_labels, val_labels, test_labels) = load_data(pickle_data_file)
    
    with tf.Session() as session: 
        session.run(tf.global_variables_initializer())
        print("Variables initialized. \n")
        
        #record keeping
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
            print('Checkpoint not found. Create new directories. Begin training.')
        checkpoint = tf.train.get_checkpoint_state(checkpoint_folder)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print('Checkpoint found. Continue training.')
        writer = tf.summary.FileWriter(graph_folder, session.graph)
        
        #train for additional num_steps steps
        initial_step = session.run(model.global_step)
        for step in range(initial_step, initial_step + num_steps):
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
    
            #one step of training
            feed_dict = {model.dataset_placeholder : batch_data, 
                         model.labels_placeholder : batch_labels,  
                         model.keep_prob_placeholder : TRAIN_KEEP_PROB}
            _, l, s = session.run([model.training_op, model.loss, model.summary_op], 
                                  feed_dict=feed_dict)
            
            #record keeping
            writer.add_summary(s, global_step=session.run(model.global_step))
        
            #evaluation on validation set
            if (step % 1000 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % session.run(model.accuracy, 
                                                                 feed_dict=feed_dict))
                feed_dict = {model.dataset_placeholder : val_dataset, 
                             model.labels_placeholder : val_labels,
                             model.keep_prob_placeholder : EVAL_KEEP_PROB}
                print("Validation accuracy: %.1f%%" % session.run(model.accuracy, 
                                                                  feed_dict=feed_dict))
                print("Learning rate: %.5f%% " % session.run(model.learning_rate))
                
                #record keeping
                saver.save(session, os.path.join(checkpoint_folder, NN_MODEL_NAME), 
                           global_step=session.run(model.global_step))
    
        #final evaluation, limit memory use by going through batches
        def batch_ix_tup_list(num_examples, batch_size):
            ix_start_list = list(map(lambda el: batch_size * el, 
                                     range((num_examples // batch_size) + 1)))
            ix_end_list = list(map(lambda el: batch_size * el, 
                                   range(1, (num_examples // batch_size) + 1)))
            ix_end_list.append(num_examples)
            return zip(ix_start_list, ix_end_list)
    
        def final_accuracy(dataset, labels, batch_size):
            num_accurate = 0
            for ix_start, ix_end in batch_ix_tup_list(labels.shape[0], batch_size):
                feed_dict = {model.dataset_placeholder : dataset[ix_start : ix_end, :], 
                             model.labels_placeholder : labels[ix_start : ix_end, :], 
                             model.keep_prob_placeholder : EVAL_KEEP_PROB}
                num_accurate += session.run(model.accuracy_count, feed_dict=feed_dict)
            return num_accurate
        
        print("\nTraining results:")
        print("Train accuracy: %.1f%%" % (100.0 * (final_accuracy(
            train_dataset, train_labels, BATCH_SIZE) / float(train_labels.shape[0]))))
        print("Validation accuracy: %.1f%%" % (100.0 * (final_accuracy(
            val_dataset, val_labels, BATCH_SIZE) / float(val_labels.shape[0]))))
        print("Test accuracy: %.1f%%" % (100.0 * (final_accuracy(
            test_dataset, test_labels, BATCH_SIZE) / float(test_labels.shape[0]))))
        
        #record keeping
        saver.save(session, os.path.join(checkpoint_folder, NN_MODEL_NAME),
                   global_step=session.run(model.global_step))
      
    
def main():
    graph = tf.Graph()
    with graph.as_default():
        nn = tf_nn.NNModel(
            topology_list=NN_TOPOLOGY_LST,
            model_name=NN_MODEL_NAME,
            start_learning_rate=START_LEARNING_RATE,
            decay_rate=DECAY_RATE,
            decay_steps=DECAY_STEPS)
        nn.build_graph()
        train(nn, PICKLE_DATA_FILE, NUM_STEPS, CHECKPOINT_FOLDER, GRAPH_FOLDER)
    
if __name__ == '__main__':
    main()