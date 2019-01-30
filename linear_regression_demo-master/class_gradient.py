from numpy import *




def Im_compute_error_from_point_function(b, m, points) :
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def Im_gradient_descent_steps_function(bias_current, derivative_current, Im_point_datasets, learning_rate):

    Im_bias = 0
    Im_derivative = 0
    Number_of_dataset = float(len(Im_point_datasets))

    for Im_iteration in range(0 , len(Im_point_datasets)):
        
        X = Im_point_datasets[Im_iteration, 0]
        Y = Im_point_datasets[Im_iteration, 1]

        bias_gradient = Im_bias - (2/Number_of_dataset * (Y - ((derivative_current * X) + bias_current)))
        derivative_gradient = Im_derivative - (2/Number_of_dataset * X * (Y - ((derivative_current * X) + bias_current)) )

    new_bias = bias_current - (learning_rate * bias_gradient)
    new_derivative = derivative_current - (learning_rate * derivative_gradient) 

    return new_bias, new_derivative





def Im_gradient_descent_function(
    Im_point_datasets, 
    initial_function_parameter_bias, 
    initial_function_parameter_derivative, 
    learning_rate, 
    Im_iteration_time): 

    Im_bias = initial_function_parameter_bias
    Im_derivative = initial_function_parameter_derivative
        
    for Im_iteration in range(Im_iteration_time):
        Im_bias, Im_derivative = Im_gradient_descent_steps_function(
            initial_function_parameter_bias, 
            initial_function_parameter_derivative,
            array(Im_point_datasets),
            learning_rate) 

    return Im_bias, Im_derivative



def Im_main_programe() :
    
    Im_point_datasets = genfromtxt("data.csv" , delimiter = ",")

    Im_learning_rate = 0.0001
    initial_function_parameter_bias = 0
    initial_function_parameter_derivative = 0 
    Im_iteration_time = 123456


    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format
        (
        initial_function_parameter_bias, 
        initial_function_parameter_derivative, 
        Im_compute_error_from_point_function
            (
            initial_function_parameter_bias, 
            initial_function_parameter_derivative, 
            Im_point_datasets
            )
        )
    )
    print ("Running...")
    Im_bias, Im_derivative = Im_gradient_descent_function(
        Im_point_datasets, 
        initial_function_parameter_bias, 
        initial_function_parameter_derivative, 
        Im_learning_rate, 
        Im_iteration_time)

    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format
    (
        Im_iteration_time, 
        Im_bias, 
        Im_derivative, 
        Im_compute_error_from_point_function
        (
            Im_bias, 
            Im_derivative, 
            Im_point_datasets)))


if __name__ == "__main__" :

    Im_main_programe()