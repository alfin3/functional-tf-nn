# functional-tf-nn

The commonly used libraries for implementing, training, and evaluating learning algorithms often improve usability at the expense of composability and research flexibility. However, a trade-off is not required to achieve a higher relational abstraction level.

The provided higher-order representation of function composition in neural networks, implemented within the constraints of a commonly used library (TensorFlow), demonstrates a segmentation of model construction with closure enabling better usability and improving composability, while providing research flexibility for modifications, such as experimenting with dropout and pruning schemes and accessing gradients.

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


