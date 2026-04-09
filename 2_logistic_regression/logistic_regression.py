from random import uniform
from math import exp
from time import sleep


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

# Initialize weights & bias to a slightly negative or slightly positive value for our SINGULAR PERCEPTRON (1 NEURON)
# Each input feature requires its own weight, so we create one for age, and one for gender
# "Can be pregnant" is also a feature, but we are using it as a target (thing to predict), so it does not get its own weight
age_weight = uniform(-.5, .5)
gender_weight = uniform(-.5, .5)
bias = uniform(-.5, .5)

# The sigmoid function is one of the several types of activation functions
# It is primarily used when targets can be binary. Either the target is Male or Female, a Dog or a Cat, 0 or 1.
# We pass in our final predicted value into the sigmoid to 
# squash our final prediction value between 0 and 1
# WITH respect to the number's polarity, and it does so in a non-linear way (for an S-shaped f(x))
# So really negative numbers obtain a very small value in the range of 0 to 1,
# and very positive numbers get a large value in the range of 0 to 1; eg -5,000,000
# The key formula is 1 / (1 + exp(-n))
# We clip the values in the range of -50 and 50 because outside of that range 
# the returned float is basically 0 or 1, and if we process numbers outside that range we 
# may get overflow errors.
def sigmoid(n) -> float:

    # sigmoid(-800) # Overflow error
    # sigmoid(800) # Overflow error

    if n < -60:
        return 0
    elif n > 60:
        return 1
    else:
        return 1 / (1 + exp(-n))

for epoch in range(epochs):

    # Initialize delta weights & bias to 0
    # These deltas, again, are magnitudes of positive or negative direction to apply to our global weights and bias
    # which are used for actual predictions, while these values below are only for keeping track of updates needed
    # based on errors
    delta_age_weight = delta_gender_weight = delta_bias = 0.0

    # Iterate through each index of data for age, gender and "can be pregnant" targets
    # You can define the result of zipping these values once outside of the loop instead of in each loop to optimize
    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):

        # Assign cbp to y for readability
        y = cbp

        # Get the weighted sum of our inputs and weights, and additionally add our bias
        # The age we're inputting gets multiplied by the age weight
        # The weight we're inputting gets multiplied by the gender weight
        # The bias tags along and is added towards the end
        # This is an example of a perception (Neuron) that has 2 weights attached to it
        # Every neuron should only ever have 1 bias in any neural network
        y_prediction = age_weight * age + gender_weight * gender + bias

        # Squash the prediction into a binary range, the closer to 1 this value is
        # the *more likely* this age and gender can be pregnant
        # the closer to 0, the *less likely* they can be pregnant
        activated_y_prediction = sigmoid(y_prediction)

        # Calculate how off we were, were we trending towards guessing "Can" or "Cannot" be pregnant? (1 or 0)
        y_error = activated_y_prediction - y

        # Lets accumulate gradients and add the contribution for this age, gender and cbp pair's error to our delta weight
        # We add the product of the error and respective input to the respective weight
        delta_age_weight += y_error * age
        delta_gender_weight += y_error * gender

        # For the bias, we simply add the error itself
        delta_bias += y_error

    # We've gone through all of our training loops, lets average the deltas by the length of inputs
    delta_age_weight /= len_data
    delta_gender_weight /= len_data
    delta_bias /= len_data

    # Now we take the product of the learning rate and the averaged, accumulated gradient 
    # and add them to the respective global weights and bias, the ones we actually use for prediction 
    # after all training loops have completed

    age_weight -= learning_rate * delta_age_weight
    gender_weight -= learning_rate * delta_gender_weight
    bias -= learning_rate * delta_bias


    # Optionally print the weights as they're updating per 1000 epochs
    if epoch % 1000 == 0:
        print(f'\rEpoch {epoch}: Age Weight: {age_weight:,.5f}; Gender Weight: {gender_weight:,.5f}, Bias: {bias:,.5f}', end='')
        # eg: Epoch 9000: Age Weight: -11.46241; Gender Weight: 8.15780, Bias: -2.59030
        sleep(0.5)


# Break out of the same-line printing
print()

print(f'Final Age Weight: {age_weight:,.5f}; Final Gender Weight: {gender_weight:,.5f}, Final Bias: {bias:,.5f}')
# Final Age Weight: -11.94072; Final Gender Weight: 8.44692, Final Bias: -2.64134

# Test with a 20 year old female, 80 year old female, 20 year old male, and 80 year old male
# We expect to see 20 yo female to be extremely likely
# Very unlikely for an 80 year old female
# Very unlikely for a 20 year old male
# Extremely unlikely for 80 year old male
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

    predicted_y = age_weight * normalized_age + gender_weight * normalized_gender + bias
    activated_y_prediction = sigmoid(predicted_y)
    cbp_predicted_y = 'Yes' if activated_y_prediction > 0.5 else 'No'
    expected_y = 'Yes' if gender == 'Female' and age < 50 else 'No'
    print(f'Prediction for {age} year old {gender}: {cbp_predicted_y} ({activated_y_prediction}); Expected: {expected_y}')
    
'''
Example Outputs for Tests
    Prediction for 20 year old Female: Yes (0.996998334575788); Expected: Yes
    Prediction for 80 year old Female: No (0.002160755951514306); Expected: No
    Prediction for 20 year old Male: No (0.06652466529443794); Expected: No
    Prediction for 80 year old Male: No (4.64614368733146e-07); Expected: No


You'll notice that a 20 year old male is more likely to be pregnant than an 80 year old female
Which biologically makes no sense, so its time to add another hidden layer
'''