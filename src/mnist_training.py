from neural_network import Model, HiddenLayer, ReLU, SoftMax
import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

keys = np.array(range(X_train.shape[0]))
np.random.shuffle(keys)
X_train = X_train[keys]
y_train = y_train[keys]


X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()

model.add(HiddenLayer(X_train.shape[1], 128))
model.add(ReLU())
model.add(HiddenLayer(128, 128))
model.add(ReLU())
model.add(HiddenLayer(128, 10))
model.add(SoftMax())

model.wrap()

model.train(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)