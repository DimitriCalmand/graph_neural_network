# Python Neural Network Library with Graph-Based Operations

Welcome to the Python Neural Network Library a powerful toolkit that leverages graph-based operations to redefine neural network development in Python. This library introduces a novel approach that eliminates the need for traditional backpropagation, providing a more efficient and intuitive way to build and train neural networks.

## Features

- **Graph-Based Operations**: Our library is founded on the principles of graph-based operations, offering an innovative alternative to traditional backpropagation. This approach simplifies the development of neural networks and enhances their efficiency.

- **Custom `g_numpy` Class**: Explore the core of our library with the custom `g_numpy` class. This class represents a significant departure from standard NumPy functionality, designed to seamlessly integrate with graph-based operations, allowing for effortless neural network construction.

- **Elegant Network Design**: Craft neural network architectures with elegance and simplicity. The graph-based approach fosters an intuitive design philosophy that makes complex network creation more accessible and streamlined.

- **Efficient Training**: With our graph-based operations, experience faster and more efficient training of your neural networks. Eliminating the need for backpropagation, our approach optimizes the training process.


## Usage

Here's how you can use the library with the custom `g_numpy` class to create and train neural networks:

```python
class Dense(layer.layer):
    def __init__(self, output_size , input_shape =  None):
        self.output_size = output_size 
        self.input_shape = input_shape
    def build(self, input):
        self.input_shape = input.shape
        self.w = self.add_trainable_array(
                (self.input_shape[1], self.output_size)
                )
        self.b = self.add_trainable_array(
                (1, self.output_size), is_bias = True
                )
    def call(self, input):
        return input @ self.w + self.b
```
## Add layers to the network
```python
model = Sequential()
model.add(Dense(3, (2,)))
```
## Compile the model
```python
def loss(y_pred, y_true):
    res = -(y_true * npg.log(y_pred) +
        (1 - y_true) * npg.log(1 - y_pred))
    res = res.mean(axis=0)
    return res
model.compile(loss)
```

## Train the model using graph-based operations
```python
model.fit(X_train, y_train, 1000, 4, verbose = 100)
```

## Make predictions
```python
model.predict(X_train)
```