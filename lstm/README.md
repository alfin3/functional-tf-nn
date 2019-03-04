# functional-tf-lstm

example of applying function composition towards building tensorflow graphs

build a multilayer LSTM from a topology list  

1) build a cell function for each layer
2) build a layer function for each layer from cell functions
3) compose layer functions into a model function 
4) evaluate the model function

higher order functions simplify code while providing research flexibility for modifications, such as rewiring cells into new variants and trying different dropout schemes. note that lambda delays evaluation, helps treating functions as first-class objects, and helps with encapsulation.

tested on English, German, and Russian text corpora that are available at: http://www.statmt.org/wmt16/translation-task.html

![layer2_unroll](https://user-images.githubusercontent.com/25671774/38122719-99067ac4-338b-11e8-8e66-ee366df8e62c.png)
