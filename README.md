# functional-tf-nn 
example of applying function composition to towards building tensorflow graphs

build a fully connected feedforward neural network of any given depth and width from a topology list 

1) define a function that builds a function for each layer
2) build layer functions
3) compose layer functions into a model function 
4) evaluate the model function

note that lambda delays evaluation, helps treating functions as first-class objects, and helps with encapsulation.

data set is available at: https://archive.ics.uci.edu/ml/datasets/Covertype (some features need pre-processing to 0 mean and unit variance).

![alt tag](https://cloud.githubusercontent.com/assets/25671774/25096616/df883790-2355-11e7-81ce-99911c271704.png)
