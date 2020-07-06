# functional-tf-nn/feedforward

An example of applying higher-order funcations towards building TensorFlow graphs.

Build a fully connected feedforward neural network of any given depth and width from a topology list:

1) build layer functions
2) compose layer functions into a model function
3) evaluate the model function

Higher order functions simplify code while providing research flexibility for modifications, such as experimenting with different dropout and pruning schemes, and accessing gradients. Note that lambda delays evaluation, helps treating functions as first-class objects, and helps with encapsulation.

The data set is available at: https://archive.ics.uci.edu/ml/datasets/Covertype (some features were pre-processed to 0 mean and unit variance).

![alt tag](https://cloud.githubusercontent.com/assets/25671774/25096616/df883790-2355-11e7-81ce-99911c271704.png)

**Quick start**

Extract `covertype.pickle`, and then run `python3 -m tf_run_nn` in an enviroment with TensorFlow (most recently tested on 1.4.0).



