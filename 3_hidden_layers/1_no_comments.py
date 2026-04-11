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

def relu(n) -> float:
    return n if n > 0 else 0

def relu_derivative(n) -> float:
    return 1.0 if n > 0 else 0.0

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

len_features = len([ages, genders])
hidden_layer_weights = [uniform(-.5, .5) for _ in range(len_features)]
hidden_layer_bias = uniform(-.5, .5)

output_layer_weight = uniform(-.5, .5)
output_layer_bias = uniform(-.5, .5)

for epoch in range(epochs):

    delta_hidden_layer_weights = [0.0 for _ in range(len_features)]
    delta_hidden_layer_bias = 0.0
    delta_output_layer_weight = 0.0
    delta_output_layer_bias = 0.0

    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):

        y = cbp

        preactivated_hidden_layer_output_value = hidden_layer_weights[0] * age + hidden_layer_weights[1] * gender + hidden_layer_bias
        activated_hidden_layer_output_value = relu(preactivated_hidden_layer_output_value)

        y_prediction = activated_hidden_layer_output_value * output_layer_weight + output_layer_bias
        activated_y_prediction = sigmoid(y_prediction)

        y_error = activated_y_prediction - y
        delta_output_layer_weight += y_error * activated_hidden_layer_output_value
        delta_output_layer_bias += y_error

        hidden_layer_error = y_error * output_layer_weight * relu_derivative(preactivated_hidden_layer_output_value)
        delta_hidden_layer_weights[0] += hidden_layer_error * age
        delta_hidden_layer_weights[1] += hidden_layer_error * gender
        delta_hidden_layer_bias += hidden_layer_error

    delta_hidden_layer_weights[0] /= len_data
    delta_hidden_layer_weights[1] /= len_data
    delta_hidden_layer_bias /= len_data

    delta_output_layer_weight  /= len_data
    delta_output_layer_bias /= len_data

    hidden_layer_weights[0] -= learning_rate * delta_hidden_layer_weights[0]
    hidden_layer_weights[1] -= learning_rate * delta_hidden_layer_weights[1]
    hidden_layer_bias -= learning_rate * delta_hidden_layer_bias

    output_layer_weight -= learning_rate * delta_output_layer_weight 
    output_layer_bias -= learning_rate * delta_output_layer_bias

    if epoch % 1000 == 0:
        print(f'\rEpoch {epoch}: Age Weight: {hidden_layer_weights[0]:,.5f}; Gender Weight: {hidden_layer_weights[1]:,.5f}; HL Bias: {hidden_layer_bias:,.5f} Output Layer Weight: {output_layer_weight:,.5f}; Output Layer Bias: {output_layer_bias:,.5f}', end='')
        sleep(0.5)

print()

print(f'Final Weights - Age Weight: {hidden_layer_weights[0]:,.5f}; Gender Weight: {hidden_layer_weights[1]:,.5f}; \ HL Bias: {hidden_layer_bias:,.5f} \
      Output Layer Weight: {output_layer_weight:,.5f}; Output Layer Bias: {output_layer_bias:,.5f}')

tests: list[tuple[int, str]] = [
    (20, 'Female'),
    (80, 'Female'),
    (20, 'Male'),
    (80, 'Male')
]

for test in tests:

    age, gender = test # type: ignore
    normalized_age = normalize_age(age)
    normalized_gender = 0 if gender == 'Male' else 1
    
    preactivated_hidden_layer_output_value = hidden_layer_weights[0] * normalized_age + hidden_layer_weights[1] * normalized_gender + hidden_layer_bias
    activated_hidden_layer_output_value = relu(preactivated_hidden_layer_output_value)

    y_prediction = activated_hidden_layer_output_value * output_layer_weight + output_layer_bias
    activated_y_prediction = sigmoid(y_prediction)          

    cbp_predicted_y = 'Yes' if activated_y_prediction > 0.5 else 'No'
    expected_y = 'Yes' if gender == 'Female' and age < 50 else 'No'
    print(f'Prediction for {age} year old {gender}: {cbp_predicted_y} ({activated_y_prediction}); Expected: {expected_y}')
    
