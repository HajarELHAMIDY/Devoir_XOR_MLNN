import numpy as np
import activation_functions
import init_methods

class MultiLayerNN:

    def __init__(self, input_dims,hidden_dims,output_dims,activation, learning_rate, num_epochs):

        self.__input_dims = input_dims
        self.__hidden_dims = hidden_dims
        self.__output_dims = output_dims
        self.__activations = [activation, 'sigmoid']
      

        self.__learning_rate = learning_rate
        self.__num_epochs = num_epochs
        self.__weights=None
        self.__bias=None

    def __init_weights(self):
      
        initial_parameters = init_methods.initialize_parameters_random(self.__input_dims,self.__hidden_dims,self.__output_dims)
      
        self.__hidden_weights,self.__hidden_bias,self.__output_weights,self.__output_bias=initial_parameters


    def __deep_forward_propagation(self, X):
        dispatcher = {
            'sigmoid': activation_functions.sigmoid,
            'relu': activation_functions.relu,
          
        }
        activation = dispatcher[self.__activations[0]]
        hidden_layer_activation = np.dot(X, self.__hidden_weights)
        hidden_layer_activation += self.__hidden_bias
        hidden_layer_output = activation(hidden_layer_activation)
        activation = dispatcher[self.__activations[1]]
        output_layer_activation = np.dot(hidden_layer_output, self.__output_weights)
        output_layer_activation += self.__output_bias
        predicted_output = activation(output_layer_activation)
        return predicted_output,hidden_layer_output

    def __deep_back_propagation(self,predicted_output,expected_output,hidden_layer_output):
        dispatcher = {
            'sigmoid': activation_functions.sigmoid_d,
            'relu': activation_functions.relu_d,
           
        }
        activation_p = dispatcher[self.__activations[0]]
        error = expected_output - predicted_output
        d_predicted_output = error * activation_p(predicted_output)
        activation_p = dispatcher[self.__activations[1]]
        error_hidden_layer = d_predicted_output.dot(self.__output_weights.T)
        d_hidden_layer = error_hidden_layer * activation_p(hidden_layer_output)
        return d_predicted_output,d_hidden_layer

    def __update_weights(self,d_predicted_output,d_hidden_layer,hidden_layer_output,X):
        self.__output_weights += hidden_layer_output.T.dot(d_predicted_output) * self.__learning_rate
        self.__output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * self.__learning_rate
        self.__hidden_weights += X.T.dot(d_hidden_layer) * self.__learning_rate
        self.__hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * self.__learning_rate

    def fit(self, X, y):
        y=y.reshape(y.shape[0],1)

        self.__init_weights()

        for _ in range(self.__num_epochs):
    
            predicted_output,hidden_layer_output=self.__deep_forward_propagation(X)
            print(predicted_output)
 
            d_predicted_output,d_hidden_layer=self.__deep_back_propagation(predicted_output,y,hidden_layer_output)

            self.__update_weights(d_predicted_output,d_hidden_layer,hidden_layer_output,X)


    def predict(self, X, threshold=0.5):
        AL, _ = self.__deep_forward_propagation(X)

        predictions = (AL >= threshold)
        predictions=predictions.reshape(1,predictions.shape[0])[0]
        return (predictions).astype(int)