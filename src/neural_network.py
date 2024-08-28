import numpy as np

# Input Layer: Just passes the input forward
class InputLayer:
    
    def forward(self, inputs, training):
        self.output = inputs


# Hidden Layer: Computes the dot product of inputs and weights and adds biases
class HiddenLayer:

    def __init__(self, num_inputs, num_neurons, 
                 weight_regularizer_1=0, weight_regularizer_2=0,
                 bias_regularizer_1=0, bias_regularizer_2=0):
        # Xavier initialization
        self.weights = np.random.randn(num_inputs, num_neurons) / np.sqrt(num_inputs)
        self.biases = np.random.randn(1, num_neurons) / np.sqrt(num_neurons)
        
        # Regularization parameters
        self.weight_regularizer_1 = weight_regularizer_1
        self.weight_regularizer_2  = weight_regularizer_2
        self.bias_regularizer_1 = bias_regularizer_1
        self.bias_regularizer_2 = bias_regularizer_2
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, deriv_values):
        # Gradients of weights and biases
        self.deriv_weights = np.dot(self.inputs.T, deriv_values)
        self.deriv_biases = np.sum(deriv_values, axis=0, keepdims=True)

        # Apply L1 regularization
        if self.weight_regularizer_1 > 0:
            l1 = np.ones_like(self.weights)
            l1[self.weights < 0] = -1
            self.deriv_weights += self.weight_regularizer_1 * l1
        
        # Apply L2 regularization
        if self.weight_regularizer_2 > 0:
            self.deriv_weights += 2 * self.weight_regularizer_2 * self.weights

        # Apply L1 regularization to biases
        if self.bias_regularizer_1 > 0:
            l1 = np.ones_like(self.biases)
            l1[self.biases < 0] = -1
            self.deriv_biases += self.bias_regularizer_1 * l1
        
        # Apply L2 regularization to biases
        if self.bias_regularizer_2 > 0:
            self.deriv_biases += 2 * self.bias_regularizer_2 * self.biases
        
        # Gradient of the loss with respect to the input
        self.deriv_inputs = np.dot(deriv_values, self.weights.T)


# ReLU Activation Layer: Applies the ReLU function
class ReLU:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, deriv_values):
        self.deriv_inputs = deriv_values.copy()
        self.deriv_inputs[self.inputs < 0] = 0


# SoftMax Activation Layer: Applies the SoftMax function
class SoftMax:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)


# Backward Gradient Loss: Simplifies SoftMax + Cross Entropy
class SoftMaxCrossEntropyLossBackward:

    def backward(self, deriv_values, y_true):
        samples = len(deriv_values)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.deriv_inputs = deriv_values.copy()
        self.deriv_inputs[range(samples), y_true] -= 1
        self.deriv_inputs = self.deriv_inputs / samples


# SGD Optimizer: Implements Stochastic Gradient Descent
class StochasticGradientDescent:

    def __init__(self, learning_rate=0.1, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0 
        self.momentum = momentum
    
    def preset_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            u_weights = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.deriv_weights
            layer.weight_momentums = u_weights

            u_biases = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.deriv_biases
            layer.bias_momentums = u_biases
        else:
            u_weights = -self.current_learning_rate * layer.deriv_weights
            u_biases = -self.current_learning_rate * layer.deriv_biases
        
        layer.weights += u_weights
        layer.biases += u_biases

    def post_update_params(self):
        self.iterations += 1

# Parent Loss Class: Common Loss Functions
class CrossEntropy:

    def regularization_loss(self):
        regularization_loss = 0

        for layer in self.layers:
            if layer.weight_regularizer_1 > 0:
                regularization_loss += layer.weight_regularizer_1 * \
                                       np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_2 > 0:
                regularization_loss += layer.weight_regularizer_2 * \
                                       np.sum(layer.weights * \
                                              layer.weights)

            if layer.bias_regularizer_1 > 0:
                regularization_loss += layer.bias_regularizer_1 * \
                                       np.sum(np.abs(layer.biases))
                
            if layer.bias_regularizer_2 > 0:
                regularization_loss += layer.bias_regularizer_2 * \
                                       np.sum(layer.biases * \
                                              layer.biases)

        return regularization_loss

    def remember_layers(self, layers):
        self.layers = layers
    
    def calculate(self, output, y, *, include_regularization=False):
        losses = self.forward(output, y)
        loss = np.mean(losses)

        self.cum_sum += np.sum(losses)
        self.cum_count += len(losses)

        if not include_regularization:
            return loss

        return loss, self.regularization_loss()

    def cum_calculate(self, *, include_regularization=False):
        loss = self.cum_sum / self.cum_count

        if not include_regularization:
            return loss
        
        return loss, self.regularization_loss()

    def reset(self):
        self.cum_sum = 0 
        self.cum_count = 0

    def forward(self, pred, true):
        samples = len(pred)
        pred = np.clip(pred, 1e-7, 1-1e-7)

        if len(true.shape) == 1:
            confidence = pred[range(samples), true]
        
        elif len(true.shape) == 2:
            confidence = np.sum(pred * true, axis=1)

        negative = -np.log(confidence)
        return negative

class CategoricalAccuracy:
    
    def calculate(self, pred, true):
        comparisons = self.compare(pred, true)

        accuracy = np.mean(comparisons)

        self.cum_sum += sum(comparisons)
        self.cum_count += len(comparisons)

        return accuracy
    
    def cum_calculate(self):
        accuracy = self.cum_sum / self.cum_count

        return accuracy

    def reset(self):
        self.cum_sum = 0
        self.cum_count = 0 

    def compare(self, pred, true):
        if len(true.shape) == 2:
            true = np.argmax(true, axis=1)
        return pred == true
    
class Model:
    
    def __init__(self):
        self.layers = []
        self.loss = CrossEntropy()
        self.optimizer = StochasticGradientDescent(decay=1e-3)
        self.accuracy = CategoricalAccuracy()
        self.softmax_classifier_output = SoftMaxCrossEntropyLossBackward()

    def add(self, layer):
        self.layers.append(layer)
    
    def wrap(self):
        self.input_layer = InputLayer()

        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss  
                self.output_layer_activation = self.layers[i]  
        
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
        
        self.loss.remember_layers(self.trainable_layers)
    
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1

            X_val, y_val = validation_data
        
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1
            
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        
        for epoch in range(1, epochs+1):
            print(f'epoch: {epoch}')

            self.loss.reset()
            self.accuracy.reset()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                
                output = self.forward(batch_X, training=True)

                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer_activation.predictions(
                                  output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)
                self.backward(output, batch_y)

                self.optimizer.preset_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')
            
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.cum_calculate(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.cum_calculate()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:

                self.loss.reset()
                self.accuracy.reset()

                for step in range(validation_steps):

                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val

                    else:
                        batch_X = X_val[
                            step*batch_size:(step+1)*batch_size
                        ]
                        batch_y = y_val[
                            step*batch_size:(step+1)*batch_size
                        ]

                    output = self.forward(batch_X, training=False)

                    self.loss.calculate(output, batch_y)

                    predictions = self.output_layer_activation.predictions(
                                      output)
                    self.accuracy.calculate(predictions, batch_y)

                validation_loss = self.loss.cum_calculate()
                validation_accuracy = self.accuracy.cum_calculate()

                print(f'validation, ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')
    
    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        return layer.output

    def backward(self, output, y):
        self.softmax_classifier_output.backward(output, y)

        self.layers[-1].deriv_inputs = self.softmax_classifier_output.deriv_inputs

        for layer in reversed(self.layers[:-1]):
            layer.backward(layer.next.deriv_inputs)
