from numpy import exp, array, random, dot

class NeuralNetwork(): 

    def __init__(self):
        # 
        self.Im_synaptic_weights = 2 * random.random((3,1)) - 1

    def __sigmoid(self, _Im_input):
        return 1 / (1 + exp(-_Im_input))

    def __sigmord_derivative_function(self, _Im_input) :
        return _Im_input * (1 - _Im_input)


    def training(self, training_set_of_input, training_set_of_output, Im_number_of_traing_iteration_times):
        for Im_iteration in range(Im_number_of_traing_iteration_times):

            #very important here (chinese back ground sounds)
            #generate the output of training inputs through neural networkprediction
            Im_predicted_output = self.prediction(training_set_of_input)
            #calculate the error values
            Im_error = training_set_of_output - Im_predicted_output

            #making adjustment by multiply the error of the inputs ad by gradient of sigmoid curve
            Im_adjustment = dot(training_set_of_input.T , Im_error * self.__sigmord_derivative_function(Im_predicted_output))

            self.Im_synaptic_weights = self.Im_synaptic_weights + Im_adjustment
            

    def prediction(self, inputs):
        #pass input through our neural network
        return self.__sigmoid(dot(inputs, self.Im_synaptic_weights))


if __name__ == "__main__" :
    
    #create a single neuron neural network
    neural_network = NeuralNetwork()

    print("The random starting synaptic weights is : ")
    print(neural_network.Im_synaptic_weights)

    #here is the training set data
    Im_training_set_inputs = array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    Im_training_set_outputs = array([[0,1,1,0]]).T

    #train the neural network using a training set.
    #repeat 10000 times and make slightly adjustment each time

    neural_network.training(Im_training_set_inputs, Im_training_set_outputs, 100000)
    print("The fresh new synaptic weights after training is : ")
    print(neural_network.Im_synaptic_weights)

    print("Predict new situation [1, 0, 0] : ")
    print(neural_network.prediction(array([1, 0, 0])))


