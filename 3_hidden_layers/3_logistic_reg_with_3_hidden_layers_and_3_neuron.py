from random import uniform
from math import exp
from time import sleep

#TODO remove repetitive comments from other code, only comment on newer code
#TODO add new comments where we didnt go over something yet

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

# Define number of epochs (we can use less with a more complex model to avoid overfitting)
epochs = 2_500


hidden_layer_1_input_size = len([ages, genders])

hidden_layer_neurons = 3

# Instead of 1 layer with 3 neurons that accept 2 input features, we create 3 layers of these, but only the first layer will contain 2 inputs
# hidden_layer_weights = [[uniform(-.5, .5) for _ in range(len_features)] for _ in range(hidden_layer_neurons)]
hidden_layers = 3
first_hidden_layer = [[[uniform(-.5, .5) for _ in range(hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(1)]

# Create a matrix with the first hidden layers weights, and the other 4 layers weights (only the first hidden layer only takes 2 inputs, the rest take 3,
# the outputs of the previous layer — each neuron has 1 output)
post_hidden_layer_1_input_size = 3
hidden_layer_weights = first_hidden_layer + [[[uniform(-.5, .5) for _ in range(post_hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers - 1)]
# Since we now have 3 hidden layers, each neuron requires its own bias for each layer
# hidden_layer_biases = [uniform(-.5, .5) for _ in range(hidden_layer_neurons)]
hidden_layer_biases = [[uniform(-.5, .5) for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]

# The output layer weights and bias setup does not change, since we still receive 3 inputs (3 weights) for the last hidden layer,
# and we still are using binary classification, so we only need 1 neuron
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
    delta_first_hidden_layer = [[[0.0 for _ in range(hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(1)]
    delta_hidden_layer_weights = delta_first_hidden_layer + [[[0.0 for _ in range(post_hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers - 1)]
    delta_hidden_layer_biases = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
    delta_output_layer_weights = [0.0 for _ in range(hidden_layer_neurons)]
    delta_output_layer_bias = 0.0

    # Iterate through each index of data for age, gender and "can be pregnant" targets
    # You can define the result of zipping these values once outside of the loop instead of in each loop to optimize
    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):

        # Assign cbp to y for readability
        y = cbp

        # Define features in an array for dynamicity later
        features = [age, gender]

        # There will STILL be three hidden layer preactivated values, we must keep track of both of them to apply activation functions separately for forward propogation
        # now and backwards propogation later. But NOW we will have 3 sets of them, the only hidden layer that accepts our age, gender, cbp values is the first hidden layer
        # The subsequent hidden layers (until the last hidden layer) receive their inputs as the previous hidden layer's output
        preactivated_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
        postactivation_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]

        # For each hidden layer
        for i in range(hidden_layers):
            # For each neuron
            for j in range(hidden_layer_neurons):
                preactivation_value = hidden_layer_biases[i][j]
                # If this is the first iteration (first hidden layer), use the data inputs (age, gender), otherwise use the previous layers outputs
                if i == 0:

                    # Define the preactivation value before weighting sums
                    preactivation_value = hidden_layer_biases[i][j]

                    # For each input for hidden layer 1
                    for k in range(hidden_layer_1_input_size):

                        preactivation_value += hidden_layer_weights[i][j][k] * features[k]
                    
                    preactivated_hidden_layer_output_values[i][j] = preactivation_value
                    postactivation_hidden_layer_output_values[i][j] = relu(preactivation_value)

                else:
                    # Calculate and save preactivated value for this neuron in array

                    # Define the preactivation value before weighting sums
                    preactivation_value = hidden_layer_biases[i][j]

                    # For each input for hidden layer 1
                    for k in range(post_hidden_layer_1_input_size):
                        # Use the previous layers post activation value as the input
                        preactivation_value += hidden_layer_weights[i][j][k] * postactivation_hidden_layer_output_values[i - 1][k]
                    
                    preactivated_hidden_layer_output_values[i][j] = preactivation_value
                    postactivation_hidden_layer_output_values[i][j] = relu(preactivation_value)

        # Calculated the weighted sum (multiply each input by its corresponding weight) of each postactivated hidden layer output per output layer weight
        # Instead of initializing "z_out" (the preactivated weighted sum after output layer processes the inputs) to 0
        # We will instead initialize it to the output layer bias we must add anyways
        z_out = output_layer_bias
        # For each neuron in the hidden layer (We use the last hidden layers postactivation values)
        for i in range(hidden_layer_neurons):
            # Add the weighted sum of the last hidden layers postactivated hidden layer outputs and respective output layer weight
            z_out += postactivation_hidden_layer_output_values[-1][i] * output_layer_weights[i]

        # The activated weighted sum is the target prediction, note we use sigmoid for binary classification (Some value between 0 "Cannot" and 1 "Can" be pregnant)
        y_prediction = sigmoid(z_out)

        '''=====> Forward Pass Ends & Backwards propagation starts here <====='''

        # y_error is how wrong we were
        y_error = y_prediction - cbp

        # calculate gradient deltas for each of the 3 output layer weights from the last hidden layers output
        for i in range(hidden_layer_neurons):
            delta_output_layer_weights[i] += y_error * postactivation_hidden_layer_output_values[-1][i]
        delta_output_layer_bias += y_error

        # Define the hidden_layer_errors for backwards propagation, this is the product of the following:
        # - y_error
        # - respective output layer weight
        # - relu derivative function of respective preactivated hidden layer output values
        hidden_layer_errors = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
        for i in range(hidden_layer_neurons):
            hidden_layer_errors[-1][i] = y_error * output_layer_weights[i] * relu_derivative(preactivated_hidden_layer_output_values[-1][i])

        for i in range(hidden_layers - 2, -1, -1):
            for j in range(hidden_layer_neurons):
                weighted_error_sum = 0.0
                for k in range(hidden_layer_neurons):
                    weighted_error_sum += hidden_layer_errors[i + 1][k] * hidden_layer_weights[i + 1][k][j]
                hidden_layer_errors[i][j] = weighted_error_sum * relu_derivative(preactivated_hidden_layer_output_values[i][j])

        # calculate the gradient deltas for each of the 2 weights in the 3 neurons
        for i in range(hidden_layers):
            for j in range(hidden_layer_neurons):
                if i == 0:
                    delta_hidden_layer_weights[i][j][0] += hidden_layer_errors[i][j] * age
                    delta_hidden_layer_weights[i][j][1] += hidden_layer_errors[i][j] * gender
                else:
                    for k in range(post_hidden_layer_1_input_size):
                        delta_hidden_layer_weights[i][j][k] += hidden_layer_errors[i][j] * postactivation_hidden_layer_output_values[i - 1][k]
                delta_hidden_layer_biases[i][j] += hidden_layer_errors[i][j]


    # We've gone through all of our training loops, lets average the deltas for each weight in each neuron by the length of our data
    # For each neuron in hidden layer:
    for i in range(hidden_layers):
        for j in range(hidden_layer_neurons):
            for k in range(len(delta_hidden_layer_weights[i][j])):
                delta_hidden_layer_weights[i][j][k] /= len_data
            delta_hidden_layer_biases[i][j] /= len_data

    # For each output layer weight
    for i in range(hidden_layer_neurons):
        delta_output_layer_weights[i] /= len_data

    # Average the lone output layer bias
    delta_output_layer_bias /= len_data

    # Start to update our weights and biases with the product of learning rate and gradients

    #For each neuron:
    for i in range(hidden_layers):
        for j in range(hidden_layer_neurons):
            for k in range(len(hidden_layer_weights[i][j])):
                hidden_layer_weights[i][j][k] -= learning_rate * delta_hidden_layer_weights[i][j][k]
            hidden_layer_biases[i][j] -= learning_rate * delta_hidden_layer_biases[i][j]

    # For each output layer weight:
    for i in range(hidden_layer_neurons):
        output_layer_weights[i] -= learning_rate * delta_output_layer_weights[i]
    
    # Update our lone output layer bias
    output_layer_bias -= learning_rate * delta_output_layer_bias



    # Optionally print the weights as they're updating per 1000 epochs
    if epoch % 1000 == 0:
        # This print statement is a bit long so we cant use the \r method to print on the same line repeatedly
        print(f'Epoch {epoch}: ' \
                f'HL Weights: {hidden_layer_weights} ' \
                f'HL Biases: {hidden_layer_biases} ' \
                f'OL Weights: {output_layer_weights} ' \
                f'OL Bias: {output_layer_bias}',
        )
        sleep(0.5)


# Break out of the same-line printing
print()

print(f'Final Weights: ' \
    f'HL Weights: {hidden_layer_weights} ' \
    f'HL Biases: {hidden_layer_biases} ' \
    f'OL Weights: {output_layer_weights} ' \
    f'OL Bias: {output_layer_bias}',
)
'''
Final Weights: 
Final Weights: 

HL Weights: [
    [
        [1.7573817820740265, -0.6307533250163528],
        [-1.263654563085226, 1.150375572954687],
        [1.0881309116397604, -0.5441669113439691]
    ],
    [
        [0.29156099572127264, -0.4060967981498772, -0.3781365723947522], 
        [-1.849050705076799, 1.7878145516860717, -1.1962173563145038],
        [0.06432261951018015, 0.2070015016401885, 0.2145205581518348]
    ], 
    [
        [-0.14717405346706292, 0.09951276022181488, -0.39754485968315245],
        [-0.3713710858123357, 0.24833428533613236, 0.07047275139610687], 
        [0.48806269908501637, 2.8290063416841362, -0.042928436637174605]
    ]
] 
HL Biases: [
    [0.3379790512963814, 0.48397212349052215, 0.36426490418920415],
    [0.02237727275561257, 0.21048404558645661, -0.46990655273549253],
    [-0.1770249069881624, -0.3903878476693229, -0.04027501440041413]
] 
OL Weights: [-0.2675430931999102, -0.21879011429978415, 2.792149103744072] 
OL Bias: -4.150273569694216
'''

def predict(age, gender):

    preactivated_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
    postactivation_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]

    # For each neuron
    for i in range(hidden_layers):
        for j in range(hidden_layer_neurons):
            if i == 0:
                # Calculate and save preactivated value for this neuron in array
                preactivated_hidden_layer_output_values[i][j] = hidden_layer_weights[i][j][0] * age + hidden_layer_weights[i][j][1] * gender + hidden_layer_biases[i][j]
            else:
                preactivation_value = hidden_layer_biases[i][j]
                for k in range(post_hidden_layer_1_input_size):
                    preactivation_value += hidden_layer_weights[i][j][k] * postactivation_hidden_layer_output_values[i - 1][k]
                preactivated_hidden_layer_output_values[i][j] = preactivation_value

            # Calculate and save postactivation value for this neuron in array
            postactivation_hidden_layer_output_values[i][j] = relu(preactivated_hidden_layer_output_values[i][j])
    
    # Calculated the weighted sum (multiply each input by its corresponding weight) of each postactivated hidden layer output per output layer weight
    # Instead of initializing "z_out" (the preactivated weighted sum after output layer processes the inputs) to 0
    # We will instead initialize it to the output layer bias we must add anyways
    z_out = output_layer_bias
    # For each neuron in the hidden layer
    for i in range(hidden_layer_neurons):
        # Add the weighted sum of the postactivated hidden layer outputs and respective output layer weight
        z_out += postactivation_hidden_layer_output_values[-1][i] * output_layer_weights[i]

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
    Prediction for 20 year old Female: Yes (0.9999999985609271); Expected: Yes
    Prediction for 80 year old Female: No (0.015515577290067317); Expected: No
    Prediction for 20 year old Male: No (0.015614777145708407); Expected: No
    Prediction for 80 year old Male: No (0.015548471709811097); Expected: No
'''