
from random import uniform
from math import exp
from time import sleep


ages = [i for i in range(20, 50)] * 2 + [i for i in range(50, 80)] * 2 
# ages = [20, 21 ... 49, 20, 21 ... 49, 50, 51 ... 79, 50, 51 ... 79]

genders = (['Female'] * 30 + ['Male'] * 30) * 2
# [30 Females, 30 Males, 30 Females, 30 Males]
# ['Female' ... 'Male' ... 'Female' ... 'Male']

# Can Be pregnant values, only the first group of Females, in the age range of 20 to 50 can be pregnant
cbps = ['Yes'] * 30 + ['No'] * 90

# Normalize values for genders and cbps (can be pregnant values), as these are discrete values, not continuous values
# and must be assigned numbers for us to perform logic on them
genders_normalized = [0 if gender == 'Male' else 1 for gender in genders]
cbps_normalized = [0 if cbp == 'No' else 1 for cbp in cbps]

def normalize_age(age) -> float:
    return (age - 20) / 60

ages_normalized = [normalize_age(age) for age in ages]

# Optionally check that all ages, genders, and cbps are the same length
assert len(set(len(x) for x in (ages, genders, cbps))) == 1, "Ages, Genders, and CBP data length mismatch"

# If asserting, safely assume the length can be the length of any of the three data sets
len_data = len(ages)

# Define a learning rate, we can go a bit higher in this logistic regression example as opposed to linear
learning_rate = 0.1

# Define number of epochs (same as last time)
epochs = 10_000


len_features = len([ages, genders])

hidden_layer_neurons = 3

# Instead of 1 neuron that accepts 2 input features, we create 3 neurons that accept 2 input features
# hidden_layer_weights = [uniform(-.5, .5) for _ in range(len_features)]
hidden_layer_weights = [[uniform(-.5, .5) for _ in range(len_features)] for _ in range(hidden_layer_neurons)]

# Since we now have 3 hidden layer neurons, each neuron requires its own bias
# hidden_layer_bias = uniform(-.5, .5)
hidden_layer_biases = [uniform(-.5, .5) for _ in range(hidden_layer_neurons)]

# Even though we know have 3 outputs from the hidden layer AKA 3 inputs to the output layer,
# we will now require 3 weights for our output neuron
# Note that The number of output neurons, not weights, is defined by the number of possible targets in a classification problem
# We are still only dealing with "Can" or "Cannot" be pregnant, so binary classification allows
# for 1 neuron. Once we start dealing with 3 or more classes, we must use that amount of neurons in the output layer
# But for now we can still use one neuron for a binary problem. Soon in this codebase we will introduce a "Maybe" value on top of "Yes" or "No"
# When we get to that point where we have 3 possible targets, we will require 3 neurons, but not here
output_layer_weights = [uniform(-.5, .5) for _ in range(hidden_layer_neurons)]
output_layer_bias = uniform(-.5, .5)

# See logistic regression example for explanation of sigmoid function
def sigmoid(n) -> float:
    if n < -60:
        return 0
    elif n > 60:
        return 1
    else:
        return 1 / (1 + exp(-n))

# Relu is an activation function 
def relu(n) -> float:
    '''Relu: If the number is greater than 0, we return the number, else 0'''
    return n if n > 0 else 0

# Relu Derivative is part of backwards propagation, you can think of it as a "backwards-activation" function
def relu_derivative(n) -> float:
    '''Relu-Deriv: If the number is greater than 0, we return 1, else 0'''
    return 1.0 if n > 0 else 0.0

for epoch in range(epochs):

    # Initialize delta weights & biases to 0
    delta_hidden_layer_weights = [[0.0 for _ in range(len_features)] for _ in range(hidden_layer_neurons)]
    delta_hidden_layer_bias = [0.0 for _ in range(hidden_layer_neurons)]
    delta_output_layer_weights = [0.0 for _ in range(hidden_layer_neurons)]
    delta_output_layer_bias = 0.0

    # Iterate through each index of data for age, gender and "can be pregnant" targets
    # You can define the result of zipping these values once outside of the loop instead of in each loop to optimize
    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):

        # Assign cbp to y for readability
        y = cbp


        # There will be three hidden layer preactivated values, we must keep track of both of them to apply activation functions separately for forward propogation
        # now and backwards propogation later
        preactivated_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]
        postactivation_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]

        # For each neuron
        for i in range(hidden_layer_neurons):
            # Calculate and save preactivated value for this neuron in array
            preactivated_hidden_layer_output_values[i] = hidden_layer_weights[i][0] * age + hidden_layer_weights[i][1] * gender + hidden_layer_biases[i]

            # Calculate and save postactivation value for this neuron in array
            postactivation_hidden_layer_output_values[i] = relu(preactivated_hidden_layer_output_values[i])
        
        # Calculated the weighted sum (multiply each input by its corresponding weight) of each postactivated hidden layer output per output layer weight
        # Instead of initializing "z_out" (the preactivated weighted sum after output layer processes the inputs) to 0
        # We will instead initialize it to the output layer bias we must add anyways
        z_out = output_layer_bias
        # For each neuron in the hidden layer
        for i in range(hidden_layer_neurons):
            # Add the weighted sum of the postactivated hidden layer outputs and respective output layer weight
            z_out += postactivation_hidden_layer_output_values[i] * output_layer_weights[i]

        # The activated weighted sum is the target prediction, note we use sigmoid for binary classification (Some value between 0 "Cannot" and 1 "Can" be pregnant)
        y_prediction = sigmoid(z_out)

        '''=====> Forward Pass Ends & Backwards propagation starts here <====='''

        # y_error is how wrong we were
        y_error = y_prediction - cbp

        # calculate gradient deltas for each of the 3 output layer weights
        for i in range(hidden_layer_neurons):
            delta_output_layer_weights[i] += y_error * postactivation_hidden_layer_output_values[i]
        delta_output_layer_bias += y_error

        # Define the hidden_layer_errors for backwards propagation, this is the product of the following:
        # - y_error
        # - respective output layer weight
        # - relu derivative function of respective preactivated hidden layer output values
        hidden_layer_errors = [0.0 for _ in range(hidden_layer_neurons)]
        for i in range(hidden_layer_neurons):
            hidden_layer_errors[i] = y_error * output_layer_weights[i] * relu_derivative(preactivated_hidden_layer_output_values[i])

        # calculate the gradient deltas for each of the 2 weights in the 3 neurons
        for i in range(hidden_layer_neurons):
            delta_hidden_layer_weights[i][0] += hidden_layer_errors[i] * age
            delta_hidden_layer_weights[i][1] += hidden_layer_errors[i] * gender
            delta_hidden_layer_bias[i] += hidden_layer_errors[i]


    # We've gone through all of our training loops, lets average the deltas for each weight in each neuron by the length of our data
    # For each neuron in hidden layer:
    for i in range(hidden_layer_neurons):
        delta_hidden_layer_weights[i][0] /= len_data
        delta_hidden_layer_weights[i][1] /= len_data
        delta_hidden_layer_bias[i] /= len_data

    # For each output layer weight
    for i in range(hidden_layer_neurons):
        delta_output_layer_weights[i] /= len_data

    # Average the lone output layer bias
    delta_output_layer_bias /= len_data

    # Start to update our weights and biases with the product of learning rate and gradients

    #For each neuron:
    for i in range(hidden_layer_neurons):
        hidden_layer_weights[i][0] -= learning_rate * delta_hidden_layer_weights[i][0]
        hidden_layer_weights[i][1] -= learning_rate * delta_hidden_layer_weights[i][1]
        hidden_layer_biases[i] -= learning_rate * delta_hidden_layer_bias[i]

    # For each output layer weight:
    for i in range(hidden_layer_neurons):
        output_layer_weights[i] -= learning_rate * delta_output_layer_weights[i]
    
    # Update our lone output layer bias
    output_layer_bias -= learning_rate * delta_output_layer_bias



    # Optionally print the weights as they're updating per 1000 epochs
    if epoch % 1000 == 0:
        # This print statement is a bit long so we cant use the \r method to print on the same line repeatedly
        print(f'Epoch {epoch}: ' \
                f'Age Weights: {[str(hidden_layer_weights[i][0])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'Gender Weights: {[str(hidden_layer_weights[i][1])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'HL Biases: {[str(hidden_layer_biases[i])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'OL Weights: {[str(output_layer_weights[i])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'OL Bias: {output_layer_bias:,.5f}',
        )
        sleep(0.5)


# Break out of the same-line printing
print()

print(f'Final Weights: ' \
        f'Age Weights: {[str(hidden_layer_weights[i][0])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'Gender Weights: {[str(hidden_layer_weights[i][1])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'HL Biases: {[str(hidden_layer_biases[i])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'OL Weights: {[str(output_layer_weights[i])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'OL Bias: {output_layer_bias:,.5f}',
)
'''
Final Weights: 
Age Weights: ['2.197022', '6.350383', '-0.10213'] 
Gender Weights: ['-1.70595', '-3.55279', '0.084955'] 
HL Biases: ['0.838651', '1.054275', '-0.19434'] 
OL Weights: ['-2.86515', '-7.33801', '-0.40031'] 
OL Bias: 5.17048
'''

def predict(age, gender):

    preactivated_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]
    postactivation_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]

    # For each neuron
    for i in range(hidden_layer_neurons):
        # Calculate and save preactivated value for this neuron in array
        preactivated_hidden_layer_output_values[i] = hidden_layer_weights[i][0] * age + hidden_layer_weights[i][1] * gender + hidden_layer_biases[i]

        # Calculate and save postactivation value for this neuron in array
        postactivation_hidden_layer_output_values[i] = relu(preactivated_hidden_layer_output_values[i])
    
    # Calculated the weighted sum (multiply each input by its corresponding weight) of each postactivated hidden layer output per output layer weight
    # Instead of initializing "z_out" (the preactivated weighted sum after output layer processes the inputs) to 0
    # We will instead initialize it to the output layer bias we must add anyways
    z_out = output_layer_bias
    # For each neuron in the hidden layer
    for i in range(hidden_layer_neurons):
        # Add the weighted sum of the postactivated hidden layer outputs and respective output layer weight
        z_out += postactivation_hidden_layer_output_values[i] * output_layer_weights[i]

    # The activated weighted sum is the target prediction, note we use sigmoid for binary classification (Some value between 0 "Cannot" and 1 "Can" be pregnant)
    y_prediction = sigmoid(z_out)
    return y_prediction
# Test with a 20 year old female, 80 year old female, 20 year old male, and 80 year old male
# We expect to see 20 yo female to be extremely likely
# Very unlikely for an 80 year old female
# Very unlikely for a 20 year old male
# Extremely unlikely for
tests: list[tuple[int, str]] = [
    (20, 'Female'),
    (80, 'Female'),
    (20, 'Male'),
    (80, 'Male')
]

for test in tests:
    age_int, gender_str = test
    normalized_age = normalize_age(age_int)
    normalized_gender = 0 if gender_str == 'Male' else 1
    
    activated_y_prediction = predict(normalized_age, normalized_gender)      
    cbp_predicted_y = 'Yes' if activated_y_prediction > 0.5 else 'No'
    expected_y = 'Yes' if gender_str == 'Female' and age_int < 50 else 'No'
    print(f'Prediction for {age_int} year old {gender_str}: {cbp_predicted_y} ({activated_y_prediction}); Expected: {expected_y}')
    
'''
Example Outputs for Tests
    Prediction for 20 year old Female: Yes (0.9999900117803036); Expected: Yes
    Prediction for 80 year old Female: No (6.788122993110865e-13); Expected: No
    Prediction for 20 year old Male: No (0.006841323372581224); Expected: No
    Prediction for 80 year old Male: No (1.806069154771185e-25); Expected: No
'''
