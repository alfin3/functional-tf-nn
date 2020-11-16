# functional-tf-nn

The commonly used libraries for implementing, training, and evaluating learning algorithms often improve usability at the expense of composability and research flexibility. However, achieving a higher relational abstraction level does not require a trade-off between usability and flexibility.

The provided examples of higher-order representation of function composition in neural networks were implemented within the constraints of a commonly used library (TensorFlow), demonstrate a segmentation of model construction with closure, and enable code simplicity and research flexibility for modifications, such as experimenting with dropout and pruning schemes, and accessing gradients. Generalizing this approach could potentially provide extensibility comparable to a language such as LISP.

# functional-tf-nn/feedforward
Build a fully connected feedforward neural network from a topology list:

1) build layer functions
2) compose layer functions into a model function
3) evaluate the model function

# functional-tf-nn/lstm
Build a multilayer LSTM from a topology list: 

1) build a cell function for each layer
2) build layer functions from cell functions
3) compose layer functions into a model function 
4) evaluate the model function


