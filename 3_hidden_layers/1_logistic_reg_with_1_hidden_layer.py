from random import uniform
from math import exp
from time import sleep

#TODO remove repetitive comments from other code, only comment on newer code
#TODO add new comments where we didnt go over something yet
'''
The intention of the next few lines is to create some "play" dataset (120 total entries), where we have
a range of ages between 20 to 80 years old, and genders that are either Male or Female.
We want a group of Males and Females for each year between 20 to 80
The "Target" in this dataset, or "Y" value will "Can Be Pregnant?" — where
the intention is to (generically) state that 
===> only females between ages 20 to 50 can be pregnant <====,
without respect to any anomalies or gender arguments.
'''
ages = [i for i in range(20, 50)] * 2 + [i for i in range(50, 80)] * 2 
# ages = [20, 21 ... 49, 20, 21 ... 49, 50, 51 ... 79, 50, 51 ... 79]

genders = (['Female'] * 30 + ['Male'] * 30) * 2
# [30 Females, 30 Males, 30 Females, 30 Males]
# ['Female' ... 'Male' ... 'Female' ... 'Male']

# Can Be pregnant values, only the first group of Females, in the age range of 20 to 50 can be pregnant
cbps = ['Yes'] * 30 + ['No'] * 90

# Normalize values for genders and cbps (can be pregnant values), as these are discrete values, not continuous values
# and must be assigned numbers for us to perform logic on them
# Discrete Data Example: ['cat', 'bird', 'dog'] 
# Continous Data Example: [100, 200, 400]
genders_normalized = [0 if gender == 'Male' else 1 for gender in genders]
# [1 for 30 times, 0 for 30 times, 1 for 30 times, 0 for 30 times]
cbps_normalized = [0 if cbp == 'No' else 1 for cbp in cbps]
# [1 for 30 times, 0 for 90 times]

# We should also normalize ages to a range of 0 to 1 for better results 
# We are normalizing with respect to the age range, not polarity, so we do not use a sigmoid function)
# We instead use a dedicated function based on: age / age range
# The min age is 20, and age range is from 20 - 80, so we use 60 as the denominator
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

'''Old Weight Setup for Logistic Regression
# Initialize weights & bias to a slightly negative or slightly positive value
#  Each input feature requires its own weight, so we create one for age, and one for gender
# "Can be pregnant" is also a feature, but we are using it as a target (thing to predict), so it does not get its own weight
# age_weight = uniform(-.5, .5)
# gender_weight = uniform(-.5, .5)
# bias = uniform(-.5, .5)
'''

# There is an "input" layer in any neural network
# However, this does not count as an actual layer, since there is no logic applied *to* that layer, since there are no weights or biases.
# The "Output" layer is a valid layer, and any additional layers will be "hidden" layers between the "input" layer and "output layer"
# 2 Layer Neural Network: INPUT LAYER* => HIDDEN LAYER => OUTPUT LAYER
# 3 Layer Neural Network: INPUT LAYER* => HIDDEN LAYER 1 => HIDDEN LAYER 2 => OUTPUT LAYER
# Lets create weights and biases for 1 hidden layer, the number of weights is defined by the number of input features
len_features = len([ages, genders])
hidden_layer_weights = [uniform(-.5, .5) for _ in range(len_features)]
hidden_layer_bias = uniform(-.5, .5)

# The outputs from the hidden layer neuron (singular) will be singular to the output layer neuron (singular)
# The Age and Gender gets passed from the Input layer* to the Hidden Layer Neuron 
# Then the Hidden Layer Neuron spits out some singular float, which goes into the output layer neuron
# This means we go from 2 features (age and gender) to 1 feature (float), so the output layer will only
# need one weight and bias
# Note that the final (or only) layer in any neural network is the output layer.
output_layer_weight = uniform(-.5, .5)
output_layer_bias = uniform(-.5 ,.5)

# See logistic regression example for explanation of sigmoid function
def sigmoid(n) -> float:
    if n < -60:
        return 0
    elif n > 60:
        return 1
    else:
        return 1 / (1 + exp(-n))

def relu(n) -> float:
    return n if n > 0 else 0

def relu_derivative(n) -> float:
    return 1.0 if n > 0 else 0.0
for epoch in range(epochs):

    # Initialize delta weights & biases to 0
    # These deltas, again, are magnitudes of positive or negative direction to apply to our global weights and biases
    # which are used for actual predictions, while these values below are only for keeping track of updates needed
    # based on errors
    delta_hidden_layer_weights = [0.0 for _ in range(len_features)]
    delta_hidden_layer_bias = 0.0
    delta_output_layer_weight = 0.0
    delta_output_layer_bias = 0.0
    # Iterate through each index of data for age, gender and "can be pregnant" targets
    # You can define the result of zipping these values once outside of the loop instead of in each loop to optimize
    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):

        # Assign cbp to y for readability
        y = cbp

        # For our hidden layer's output:
        # Get the weighted sum of our inputs and weights, and additionally add our bias
        # The age we're inputting gets multiplied by the age weight
        # The weight we're inputting gets multiplied by the gender weight
        # The bias tags along and is added towards the end
        # This is an example of a perception (Neuron) that has 2 weights attached to it
        # Every neuron should only ever have 1 bias in any neural network
        preactivated_hidden_layer_output_value = hidden_layer_weights[0] * age + hidden_layer_weights[1] * gender + hidden_layer_bias
        activated_hidden_layer_output_value = relu(preactivated_hidden_layer_output_value)

        # Use the hidden layer output with our output layer weight and bias
        y_prediction = activated_hidden_layer_output_value * output_layer_weight + output_layer_bias

        # Squash the prediction into a binary range, the closer to 1 this value is
        # the *more likely* this age and gender can be pregnant
        # the closer to 0, the *less likely* they can be pregnant
        activated_y_prediction = sigmoid(y_prediction)

        # Calculate how off we were, were we trending towards guessing "Can" or "Cannot" be pregnant? (1 or 0)
        y_error = activated_y_prediction - y

        delta_output_layer_weight += y_error * activated_hidden_layer_output_value
        delta_output_layer_bias += y_error

        hidden_layer_error = y_error * output_layer_weight * relu_derivative(preactivated_hidden_layer_output_value)
        delta_hidden_layer_weights[0] += hidden_layer_error * age
        delta_hidden_layer_weights[1] += hidden_layer_error * gender
        delta_hidden_layer_bias += hidden_layer_error

    # We've gone through all of our training loops, lets average the deltas by the length of inputs
    delta_hidden_layer_weights[0] /= len_data
    delta_hidden_layer_weights[1] /= len_data
    delta_hidden_layer_bias /= len_data

    delta_output_layer_weight  /= len_data
    delta_output_layer_bias /= len_data

    # Now we take the product of the learning rate and the averaged, accumulated gradient 
    # and add them to the respective global weights and bias, the ones we actually use for prediction 
    # after all training loops have completed


    hidden_layer_weights[0] -= learning_rate * delta_hidden_layer_weights[0]
    hidden_layer_weights[1] -= learning_rate * delta_hidden_layer_weights[1]
    hidden_layer_bias -= learning_rate * delta_hidden_layer_bias

    output_layer_weight -= learning_rate * delta_output_layer_weight 
    output_layer_bias -= learning_rate * delta_output_layer_bias


    # Optionally print the weights as they're updating per 1000 epochs
    if epoch % 1000 == 0:
        print(f'\rEpoch {epoch}: Age Weight: {hidden_layer_weights[0]:,.5f}; Gender Weight: {hidden_layer_weights[1]:,.5f}; HL Bias: {hidden_layer_bias:,.5f} Output Layer Weight: {output_layer_weight:,.5f}; Output Layer Bias: {output_layer_bias:,.5f}', end='')
        # eg: Epoch 9000: Age Weight: -11.46241; Gender Weight: 8.15780, Bias: -2.59030
        sleep(0.5)


# Break out of the same-line printing
print()

print(f'Final Weights - Age Weight: {hidden_layer_weights[0]:,.5f}; Gender Weight: {hidden_layer_weights[1]:,.5f}; HL Bias: {hidden_layer_bias:,.5f} Output Layer Weight: {output_layer_weight:,.5f}; Output Layer Bias: {output_layer_bias:,.5f}')
# Final Age Weight: -11.96521; Final Gender Weight: 8.46045, Final Bias: -2.64272

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
    age, gender = test
    normalized_age = normalize_age(age)
    normalized_gender = 0 if gender == 'Male' else 1
    
    preactivated_hidden_layer_output_value = hidden_layer_weights[0] * normalized_age + hidden_layer_weights[1] * normalized_gender + hidden_layer_bias
    activated_hidden_layer_output_value = relu(preactivated_hidden_layer_output_value)
    y_prediction = activated_hidden_layer_output_value * output_layer_weight + output_layer_bias
    activated_y_prediction = sigmoid(y_prediction)          
    cbp_predicted_y = 'Yes' if activated_y_prediction > 0.5 else 'No'
    expected_y = 'Yes' if gender == 'Female' and age < 50 else 'No'
    print(f'Prediction for {age} year old {gender}: {cbp_predicted_y} ({activated_y_prediction}); Expected: {expected_y}')
    
'''
Example Outputs for Tests
    Prediction for 20 year old Female: Yes (0.9970344616038429); Expected: Yes
    Prediction for 80 year old Female: No (0.002134292939560123); Expected: No
    Prediction for 20 year old Male: No (0.06643885264314017); Expected: No
    Prediction for 80 year old Male: No (4.5274584139804633e-07); Expected: No


You'll notice that a 20 year old male is more likely to be pregnant than an 80 year old female
Which biologically makes no sense, so its time to add another hidden layer
'''