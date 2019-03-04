"""
tf_data.py

Implements handling text data across different languages. 

Python 3.5.2
TensorFlow 1.1.0
Includes a few code snippets from CS 20SI at http://web.stanford.edu/class/cs20si/ and udacity 
repository at https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity.
"""
import numpy as np

class DATA(object):
    """Build a class for handling text data across different languages."""
    def __init__(self, language_chars, 
                 train_text, valid_text, 
                 train_unrollings, train_batch_size, 
                 valid_num_unrollings, valid_batch_size):
        self.language_chars = language_chars 
        self.train_text = train_text 
        self.valid_text = valid_text 
        self.train_unrollings = train_unrollings 
        self.train_batch_size = train_batch_size 
        self.valid_num_unrollings = valid_num_unrollings 
        self.valid_batch_size = valid_batch_size
        self.vocab_size = len(self.language_chars) + 1
        self.char2id, self.id2char = build_vocab_fn(chars=self.language_chars)
 
    def train_batches_gen(self, steps_completed):
        """Build a generator of training batches."""
        return batches_gen(steps_completed=steps_completed,
                           num_unrollings=self.train_unrollings, 
                           batch_size=self.train_batch_size, 
                           vocab_size=self.vocab_size, 
                           text=self.train_text, 
                           fn_char2id=self.char2id)
        
    def valid_batches_gen(self, steps_completed):
        """Build a generator of validation batches."""
        return batches_gen(steps_completed=steps_completed,
                           num_unrollings=self.valid_num_unrollings, 
                           batch_size=self.valid_batch_size, 
                           vocab_size=self.vocab_size, 
                           text=self.valid_text, 
                           fn_char2id=self.char2id)
    
    def random_char_probs(self):
        """Generate a probability distribution of a random character."""
        values = np.random.uniform(0.0, 1.0, size=[1, self.vocab_size]) 
        return values / np.sum(values)
    
    def sample_char_probs(self, probs=None):
        """Sample a character from a probability distribution and return 
        its one-hot-encoding."""
        if probs is None:
            probs = self.random_char_probs()
        sample_char_probs = np.zeros(shape=[1, self.vocab_size], dtype=np.float)
        ix = np.random.choice(self.vocab_size, 1, p=probs[0])
        sample_char_probs[0, ix] = 1.0
        return sample_char_probs
        
    def probs2chars(self, batch):
        """Convert batch probabilities to characters."""
        return [self.id2char(char_id) for char_id in np.argmax(batch, 1)]
        
    def batches2strings(self, batches):
        """Convert batches to strings."""
        s = [''] * batches[0].shape[0]
        for batch in batches:
            s = [''.join(tup) for tup in zip(s, self.probs2chars(batch))]
        return s
        
    def batch_logprob(self, batch, labels_batch):
        """Compute the average log-probability of the true labels in a batch,
        given one-hot-encoded labels."""
        batch[batch < 1e-10] = 1e-10
        true_labels_logprobs = np.multiply(labels_batch, -np.log(batch))
        return np.sum(true_labels_logprobs) / labels_batch.shape[0]
                
def build_vocab_fn(chars):
    """Build functions for mapping characters to integer ids and vice versa."""
    vocab_dict = dict()
    for char in chars:
        vocab_dict[char] = len(vocab_dict) + 1
    reverse_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys())) 
    def char2id(char):
        if char in chars:
            return vocab_dict[char]
        else:
            return 0
    def id2char(i):
        if i > 0 and i < len(chars) + 1:
            return reverse_vocab_dict[i]
        else:
            return ' '
    return char2id, id2char
                       
def batches_gen(steps_completed, num_unrollings, batch_size, vocab_size, text, fn_char2id):
    """Build a generator of batches."""
    def batch_gen(batch_size, vocab_size, text, fn_char2id):
        """Build a generator of a batch."""
        text_size = len(text)
        segment_size = text_size // batch_size
        cursor_pos_lst = [(ix * segment_size + 
                           steps_completed * num_unrollings) % text_size
                          for ix in range(batch_size)]
        while True:
            new_batch = np.zeros((batch_size, vocab_size))
            for ix in range(batch_size):
                new_batch[ix, fn_char2id(text[cursor_pos_lst[ix]])] = 1.0
                cursor_pos_lst[ix] = (cursor_pos_lst[ix] + 1) % text_size
            yield new_batch
    batch = batch_gen(batch_size, vocab_size, text, fn_char2id)
    last_batch = next(batch)
    while True:
        batches = [last_batch]
        for ix in range(num_unrollings):
            batches.append(next(batch))
        last_batch = batches[-1]
        yield batches          
        