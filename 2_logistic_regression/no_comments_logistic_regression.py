from random import uniform
from math import exp
from time import sleep

def normalize_age(age) -> float:
    return (age - 20) / 60

def sigmoid(n) -> float:
    if n < -60:
        return 0
    elif n > 60:
        return 1
    else:
        return 1 / (1 + exp(-n))
    
ages = [i for i in range(20, 50)] * 2 + [i for i in range(50, 80)] * 2 
ages_normalized = [normalize_age(age) for age in ages]

genders = (['Female'] * 30 + ['Male'] * 30) * 2
genders_normalized = [0 if gender == 'Male' else 1 for gender in genders]

cbps = ['Yes'] * 30 + ['No'] * 90
cbps_normalized = [0 if cbp == 'No' else 1 for cbp in cbps]

assert len(set(len(x) for x in (ages, genders, cbps))) == 1, "Ages, Genders, and CBP data length mismatch"

len_data = len(ages)

learning_rate = 0.1
epochs = 10_000

age_weight = uniform(-.5, .5)
gender_weight = uniform(-.5, .5)
bias = uniform(-.5, .5)

for epoch in range(epochs):

    delta_age_weight = delta_gender_weight = delta_bias = 0.0

    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):

        y = cbp

        y_prediction = age_weight * age + gender_weight * gender + bias
        activated_y_prediction = sigmoid(y_prediction)

        y_error = activated_y_prediction - y
        delta_age_weight += y_error * age
        delta_gender_weight += y_error * gender
        delta_bias += y_error

    delta_age_weight /= len_data
    delta_gender_weight /= len_data
    delta_bias /= len_data

    age_weight -= learning_rate * delta_age_weight
    gender_weight -= learning_rate * delta_gender_weight
    bias -= learning_rate * delta_bias

    if epoch % 1000 == 0:
        print(f'\rEpoch {epoch}: Age Weight: {age_weight:,.5f}; Gender Weight: {gender_weight:,.5f}, Bias: {bias:,.5f}', end='')
        sleep(0.5)


print()

print(f'Final Age Weight: {age_weight:,.5f}; Final Gender Weight: {gender_weight:,.5f}, Final Bias: {bias:,.5f}')

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
    
