import numpy as np
import random

class NN: 
    def __init__(self, layers_sizes, activation = 'sigmoid'):
        ''' 
            (List(Integer), String) --> None
            layers_sizes represent the number of nodes in each layer of the neural network.  
            The program assumes that the first layer is the input layer and the last one is the output layer.
            By default, it assumes a sigmoid activation function.
        '''
        self.layers_sizes = layers_sizes
        #By convention, there are no bias units for the input layer
        self.biases = [np.random.randn(i, 1) for i in layers_sizes[1:]]
        self.weights = [np.random.randn(layers_sizes[i], layers_sizes[i-1]) for i in range(1, len(layers_sizes))]
        self.function_activation = activation

    def forward_prop(self, input_layer):
        '''
            (List(Float)) --> List(Float)
            Takes in the input layer and propogates it through the network and returns the output layer.
        '''
        output_layer = input_layer
        for i, j in zip(self.weights, self.biases):
            output_layer = self.activation_function(np.dot(i, output_layer)+j)
        return output_layer

    def activation_function(self, z):
        '''
            (List(Float)) --> List(Float)
            Returns the activation function calculated on the given vector.
        '''
        if(self.function_activation == 'sigmoid'):
            return 1.0 / (1.0+np.exp(-z))

    def activation_function_derivative(self, z):
        '''
            (List(Float)) --> List(Float)
            Returns the derivative of the activation function calculated on the given vector.
        '''
        if(self.function_activation == 'sigmoid'):
            return self.activation_function(z)*(1-self.activation_function(z))

    def train(self, training_data, epochs, batch_size, learning_rate, test_data = None):
        '''
            (List(Tuple), Integer, Integer, Float, List(Tuple) --> None
            Performs Stochastic Gradient Descent in order to train the network.  
            training_data and test_data are list of tuples containing the training inputs and the desired output.
            If test_data is provided, it will evaluate the accuracy of the network on it for every epoch.
        '''
        for i in range(epochs):
            random.shuffle(training_data)
            #creating a bunch of small batches of the training data
            batches = [training_data[j:j+batch_size] for  j in range(0, len(training_data), batch_size)]
            for k in batches:
                self.update_batch(k, learning_rate)
            if test_data:
                print("Epoch {} : {} / {}".format(i, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch "+str(i))

    def update_batch(self, batch, learning_rate):
        '''
            (List(Tuple), Float) --> None
            Updates the weights and biases using gradient descent and backpropogation from the small batch.
        '''
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]

        for x, y in batch:
            delta_gradient_weights, delta_gradient_biases = self.backprop(x, y)
            gradient_weights = [i+j for i, j in zip(gradient_weights, delta_gradient_weights)]
            gradient_biases = [i+j for i, j in zip(gradient_biases, delta_gradient_biases)]

        self.weights = [w-(learning_rate/len(batch))*nw for w, nw in zip(self.weights, gradient_weights)]
        self.biases = [b-(learning_rate/len(batch))*nb for b, nb in zip(self.biases, gradient_biases)]

    def backprop(self, x, y):
        '''
            (List(Float), List(Float)) --> Tuple(List, List)
            Returns a tuple the gradient of the cost function with respect to the weights and biases
            for each of the layers in the network.
        '''
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        
        activation = x
        activations = [x]
        z_vectors = []
        for i, j in zip(self.weights, self.biases):
            z = np.dot(i, activation) + j
            z_vectors.append(z)
            activation = self.activation_function(z)
            activations.append(activation)
        
        delta = self.delta_output(activations[-1], y) * self.activation_function_derivative(z_vectors[-1])
        gradient_weights[-1] = np.dot(delta, activations[-2].transpose())
        gradient_biases[-1] = delta

        for i in range(2, len(self.layers_sizes)):
            z = z_vectors[-i]
            derivative = self.activation_function_derivative(z)
            delta = np.dot(self.weights[-i+1].transpose(), delta) * derivative
            gradient_weights[-i] = np.dot(delta, activations[-i-1].transpose())
            gradient_biases[-i] = delta
        
        return (gradient_weights, gradient_biases)

    def delta_output(self, output_layer, y):
        '''
            (List(Float), List(Float)) --> List(Float)
            Calculates the error between the calculated output and the desired output.
        '''
        return (output_layer-y)
    
    def evaluate(self, test_data):
        '''
            (List(Tuple)) --> Int
            Taking in the test_data, it returns the amount of times the network correctly evaluates the test data.
        '''
        test_result = [(np.argmax(self.forward_prop(x)), y) for (x, y) in test_data]
        return sum((x == y) for (x, y) in test_result)

