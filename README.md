![image](https://github.com/user-attachments/assets/8a0c3e18-01d6-4ade-b157-fca7ffd5b3b0)

![Figure_1](https://github.com/user-attachments/assets/04030194-ba19-4412-906b-5dd5416e5ee5)


# Neural Network from Scratch

This project implements a basic feedforward neural network using only NumPy, with functionalities for training on datasets like MNIST. It includes core concepts like ReLU, SoftMax, regularization, and backpropagation, and supports stochastic gradient descent optimization.

---

## Features

- **Input Layer**: Passes the raw input forward.
- **Hidden Layer**: Computes the dot product of inputs and weights, adds biases, and supports L1/L2 regularization.
- **ReLU Activation**: Implements the ReLU activation function.
- **SoftMax Activation**: Implements the SoftMax activation function for classification.
- **SGD Optimizer**: Stochastic Gradient Descent with learning rate decay and momentum.
- **Cross Entropy Loss**: Computes the categorical cross-entropy loss.
- **Accuracy Tracking**: Tracks training and validation accuracy over epochs.

---

## Mathematical Concepts

### ReLU (Rectified Linear Unit)
$$
f(x) = \text{max}(0, x)
$$

### SoftMax
$$
S(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

### Cross-Entropy Loss
For true labels \( y \) and predictions \( \hat{y} \):
$$
L = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})
$$

### Weight Regularization
- **L1 Regularization**:
$$
L1 = \lambda \sum_{i,j} |W_{i,j}|
$$
- **L2 Regularization**:
$$
L2 = \lambda \sum_{i,j} W_{i,j}^2
$$

### Stochastic Gradient Descent (SGD) Update
- Without momentum:
$$
W = W - \eta \cdot \frac{\partial L}{\partial W}
$$
- With momentum:
$$
v_t = \gamma v_{t-1} + \eta \cdot \frac{\partial L}{\partial W}, \quad W = W - v_t
$$

---

## Generalized Usage

### Dataset Preprocessing
1. Normalize input data to the range \([-1, 1]\).
2. One-hot encode the labels if necessary.

```python
X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5
```

### Model Creation
```python
from neural_network import Model, HiddenLayer, ReLU, SoftMax

model = Model()
model.add(HiddenLayer(input_size, 128))
model.add(ReLU())
model.add(HiddenLayer(128, 128))
model.add(ReLU())
model.add(HiddenLayer(128, num_classes))
model.add(SoftMax())
model.wrap()
```

### Training
```python
model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
```

### Plotting Loss and Accuracy
```python
import matplotlib.pyplot as plt

plt.plot(cost_wrt, label='Loss')
plt.plot(accuracy_wrt, label='Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.legend()
plt.title('Training Progress')
plt.show()
```

---

## Example Application

Train a neural network on the MNIST dataset:

```python
from neural_network import Model, HiddenLayer, ReLU, SoftMax
import tensorflow as tf
import numpy as np

# Load and preprocess MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Initialize and train the model
model = Model()
model.add(HiddenLayer(X_train.shape[1], 128))
model.add(ReLU())
model.add(HiddenLayer(128, 128))
model.add(ReLU())
model.add(HiddenLayer(128, 10))
model.add(SoftMax())
model.wrap()

cost_wrt, accuracy_wrt = model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
```

---

## Notes

- **Batch Size**: Adjust for memory constraints.
- **Learning Rate**: Fine-tune for faster convergence.
- **Regularization**: Use L1/L2 regularization to prevent overfitting.

---

## License
This project is licensed under the MIT License.

---

Enjoy exploring and building neural networks from scratch!
