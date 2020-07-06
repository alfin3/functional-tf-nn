# functional-tf-nn/lstm

An example of applying higher-order functions towards building TensorFlow graphs.

Build a multilayer LSTM from a topology list: 

1) build a cell function for each layer
2) build layer functions from cell functions
3) compose layer functions into a model function 
4) evaluate the model function

Higher order functions simplify code while providing research flexibility for modifications, such as rewiring cells into new variants, experimenting with different dropout and pruning schemes, and accessing gradients. Note that lambda delays evaluation, helps treating functions as first-class objects, and helps with encapsulation.

Tested on English, German, and Russian text corpora that are available at: http://www.statmt.org/wmt16/translation-task.html

![layer2_unroll](https://user-images.githubusercontent.com/25671774/38122719-99067ac4-338b-11e8-8e66-ee366df8e62c.png)

**Quick start**

Extract `news-commentary-v11.en`, and run `python3 -m tf_run_char_lstm` in an enviroment with TensorFlow (most recently tested on 1.4.0).
