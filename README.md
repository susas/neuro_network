This is a pure standard c++ implementation of basic fully connected neuralnetwork, using activation function of sigmoid.
Its training algorithm works by wiggling each of the models weight by some epsilon and thus calculating and applying the gradient.

It calculates the error by the sqaure of differences.

This project includes somewhat extensive matrix class, which has multiple functionalities including:
- Matrix concatenation (horizontal and vertical)
- Matrix transposing
- Matrix multiplication

The neuronnetwork class is built on the aforementioned matrix class. Neuronetwork class the following functionalities:
- Training of the model
- Forwarding of the model

This project has bunch of areas of improvement, which include but are not limited to:
- Seperating classes in to their own implementation files.
- Backpropogation by derivatives.
- Moving activation functions to the neuralnetwork model.
- Dangling in to other types of neuralnetworks, for example convolutional neuralnetworks.
- Adding other activation functions.

This project was created as a personal exercise.