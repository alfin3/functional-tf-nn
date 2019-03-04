# functional-tf-nn

examples of applying higher-order functions towards building tensorflow graphs

higher order functions enable code simplicity while providing research flexibility for modifications, such as experimenting with different dropout and pruning schemes, and accessing gradients.

functional-tf-nn/feedforward: build a fully connected feedforward neural network of any given depth and width from a topology list

1) build layer functions
2) compose layer functions into a model function
3) evaluate the model function

