"""
tf_run_char_lstm.py

Implements training and evaluation functions.

Tested on English, German, and Russian text corpora. 

Python 3.5.2
TensorFlow 1.1.0
Includes a few code snippets from CS 20SI at http://web.stanford.edu/class/cs20si/ and udacity 
repository at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity.
"""
import os
import numpy as np
import tensorflow as tf
from tf_data import DATA
from tf_multilayer_char_lstm import CHAR_LSTM

#language
DE_LOWERCASE = 'abcdefghijklmnopqrstuvwxyzüäöß'
DE_UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÜÄÖ'
RU_LOWERCASE = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
RU_UPPERCASE = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
EN_LOWERCASE = 'abcdefghijklmnopqrstuvwxyz'
EN_UPPERCASE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LANGUAGE_CHARS = DE_LOWERCASE + DE_UPPERCASE
VOCAB_SIZE = len(LANGUAGE_CHARS) + 1

#data
EN_DATA_FILE = 'news.2015.en.shuffled'
DE_DATA_FILE = 'news.2015.de.shuffled'
RU_DATA_FILE = 'news.2015.ru.shuffled'
DATA_ROOT = './'
DATA_FILE = os.path.join(DATA_ROOT, DE_DATA_FILE)
VALID_SET_SIZE = 10000

#model
BATCH_SIZE = 256
NUM_UNROLLINGS = 10
START_LEARNING_RATE = 2.0 
DECAY_STEPS = 5000
DECAY_RATE = 0.75 
CLIP_NORM = 2.5 
TRAIN_KEEP_PROB = 0.85
EVAL_KEEP_PROB = 1.0
LSTM_TOPOLOGY = [VOCAB_SIZE, 128, 128]
CLASSIFIER_TOPOLOGY = [128, 1024, 1024, 1024, #1024, #1024, #1024, #1024
                       VOCAB_SIZE] 
NUM_STEPS = 10001
PRINT_FREQ = 100
FREQ_MULTIPLE = 10  # PRINT_FREQ mutiple for validation
PRINT_SENTENCE_SIZE = 100
PRINT_NUM_SENTENCES = 10

#records
MODEL_NAME = 'de_64_2x1024d_2'
RECORDS_ROOT = './'
CHECKPOINT_FOLDER = os.path.join(RECORDS_ROOT, 'checkpoints', MODEL_NAME, '')
GRAPH_FOLDER = os.path.join(RECORDS_ROOT, 'graphs', MODEL_NAME, '')

def read_text_file(text_file):
    """Read a text file."""
    try:
        with open(text_file, "rb") as f:
            text = tf.compat.as_str(f.read())    
    except Exception as e:
        print("Unable to open", text_file, ":", e)
        raise
    return text

def train(model, data_file, num_steps, checkpoint_folder, graph_folder):
    """The main training function."""
    saver = tf.train.Saver() 
    text = read_text_file(data_file)
    data = DATA(language_chars=LANGUAGE_CHARS, 
                train_text=text[VALID_SET_SIZE :], 
                valid_text=text[: VALID_SET_SIZE], 
                train_unrollings=NUM_UNROLLINGS, 
                train_batch_size=BATCH_SIZE, 
                valid_num_unrollings=1, 
                valid_batch_size=1)
    
    with tf.Session() as session: 
        session.run(tf.global_variables_initializer())        
        print("Variables initialized. \n")
    
        # record keeping
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
            print('Checkpoint not found. Create new directories. Begin training. \n')
        checkpoint = tf.train.get_checkpoint_state(checkpoint_folder)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(session, checkpoint.model_checkpoint_path)
            print('Checkpoint found. Continue training. \n')
        writer = tf.summary.FileWriter(graph_folder, session.graph)
    
        # train for additional num_steps steps
        steps_completed = session.run(model.global_step)
        train_batches_gen = data.train_batches_gen(steps_completed)
        mean_loss = 0.0
        for step in range(steps_completed, steps_completed + num_steps):
            
            # one step of training
            train_batches = next(train_batches_gen)
            feed_dict = dict()
            for i in range(NUM_UNROLLINGS + 1):
                feed_dict[model.train_data[i]] = train_batches[i]
            feed_dict[model.keep_prob] = TRAIN_KEEP_PROB
            _, l, train_prediction, s = session.run([model.optimizer, 
                                                     model.loss, 
                                                     model.train_prediction, 
                                                     model.summary], feed_dict=feed_dict)
            mean_loss += l
            
            # record keeping
            writer.add_summary(s, global_step=session.run(model.global_step))
            
            # evaluation 
            if step % PRINT_FREQ == 0:
                
                # minibatch train perplexity
                if step > 0:
                    mean_loss = mean_loss / PRINT_FREQ
                learning_rate = session.run(model.learning_rate)
                labels = np.concatenate(list(train_batches)[1:], axis=0)
                train_perplexity = float(np.exp(data.batch_logprob(train_prediction, labels)))
                
                # validation set perplexity
                valid_logprob = 0
                valid_batches_gen = data.valid_batches_gen(0)
                session.run(model.reset_saved_val_vars)
                for _ in range(VALID_SET_SIZE):
                    valid_batches = next(valid_batches_gen)
                    valid_prediction = session.run(model.val_prediction,
                        {model.val_input: valid_batches[0], model.keep_prob: EVAL_KEEP_PROB})
                    valid_logprob = valid_logprob + data.batch_logprob(valid_prediction, 
                                                                       valid_batches[1])
                valid_perplexity = float(np.exp(valid_logprob / VALID_SET_SIZE))
                print('Average loss and learning rate at step %d: %f, %f' % 
                      (step, mean_loss, learning_rate))
                print('Minibatch perplexity: %.2f' % train_perplexity)
                print('Validation set perplexity: %.2f' % valid_perplexity)
                mean_loss = 0
              
                # generate sentences for visual validation
                if step % (PRINT_FREQ * FREQ_MULTIPLE) == 0:
                    print('+' * (PRINT_SENTENCE_SIZE + 1))
                    for _ in range(PRINT_NUM_SENTENCES):
                        char_probs = data.sample_char_probs()
                        sentence = data.probs2chars(char_probs)[0]
                        session.run(model.reset_saved_val_vars)
                        for _ in range(PRINT_SENTENCE_SIZE):
                            char_prediction = model.val_prediction.eval(
                                {model.val_input: char_probs, model.keep_prob: EVAL_KEEP_PROB})
                            char_probs = data.sample_char_probs(char_prediction)
                            sentence += data.probs2chars(char_probs)[0]
                        print(sentence)
                    print('+' * (PRINT_SENTENCE_SIZE + 1))
                    
                #record keeping 
                saver.save(session, os.path.join(checkpoint_folder, MODEL_NAME), 
                           global_step=session.run(model.global_step))
                
def main():
    graph = tf.Graph()
    with graph.as_default():
        model = CHAR_LSTM(batch_size=BATCH_SIZE, 
                          vocab_size=VOCAB_SIZE, 
                          num_unrollings=NUM_UNROLLINGS,
                          lstm_topology=LSTM_TOPOLOGY,
                          classifier_topology=CLASSIFIER_TOPOLOGY,
                          start_learning_rate=START_LEARNING_RATE, 
                          decay_steps=DECAY_STEPS,
                          decay_rate=DECAY_RATE,
                          clip_norm=CLIP_NORM)
        model.build_graph()
        train(model, DATA_FILE, NUM_STEPS, CHECKPOINT_FOLDER, GRAPH_FOLDER)
    
if __name__ == '__main__':
    main()
