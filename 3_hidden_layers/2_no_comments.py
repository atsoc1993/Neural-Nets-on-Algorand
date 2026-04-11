from random import uniform
from math import exp
from time import sleep

ages = [i for i in range(20, 50)] * 2 + [i for i in range(50, 80)] * 2

genders = (['Female'] * 30 + ['Male'] * 30) * 2

cbps = ['Yes'] * 30 + ['No'] * 90

genders_normalized = [0 if gender == 'Male' else 1 for gender in genders]
cbps_normalized = [0 if cbp == 'No' else 1 for cbp in cbps]

def normalize_age(age) -> float:
    return (age - 20) / 60

ages_normalized = [normalize_age(age) for age in ages]

assert len(set(len(x) for x in (ages, genders, cbps))) == 1, "Ages, Genders, and CBP data length mismatch"

len_data = len(ages)

learning_rate = 0.1

epochs = 10_000

len_features = len([ages, genders])

hidden_layer_neurons = 3

hidden_layer_weights = [[uniform(-.5, .5) for _ in range(len_features)] for _ in range(hidden_layer_neurons)]

hidden_layer_biases = [uniform(-.5, .5) for _ in range(hidden_layer_neurons)]

output_layer_weights = [uniform(-.5, .5) for _ in range(hidden_layer_neurons)]
output_layer_bias = uniform(-.5, .5)

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
    delta_hidden_layer_weights = [[0.0 for _ in range(len_features)] for _ in range(hidden_layer_neurons)]
    delta_hidden_layer_bias = [0.0 for _ in range(hidden_layer_neurons)]
    delta_output_layer_weights = [0.0 for _ in range(hidden_layer_neurons)]
    delta_output_layer_bias = 0.0

    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):
        y = cbp

        preactivated_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]
        postactivation_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]

        for i in range(hidden_layer_neurons):
            preactivated_hidden_layer_output_values[i] = hidden_layer_weights[i][0] * age + hidden_layer_weights[i][1] * gender + hidden_layer_biases[i]
            postactivation_hidden_layer_output_values[i] = relu(preactivated_hidden_layer_output_values[i])

        z_out = output_layer_bias
        for i in range(hidden_layer_neurons):
            z_out += postactivation_hidden_layer_output_values[i] * output_layer_weights[i]

        y_prediction = sigmoid(z_out)

        y_error = y_prediction - cbp

        for i in range(hidden_layer_neurons):
            delta_output_layer_weights[i] += y_error * postactivation_hidden_layer_output_values[i]
        delta_output_layer_bias += y_error

        hidden_layer_errors = [0.0 for _ in range(hidden_layer_neurons)]
        for i in range(hidden_layer_neurons):
            hidden_layer_errors[i] = y_error * output_layer_weights[i] * relu_derivative(preactivated_hidden_layer_output_values[i])

        for i in range(hidden_layer_neurons):
            delta_hidden_layer_weights[i][0] += hidden_layer_errors[i] * age
            delta_hidden_layer_weights[i][1] += hidden_layer_errors[i] * gender
            delta_hidden_layer_bias[i] += hidden_layer_errors[i]

    for i in range(hidden_layer_neurons):
        delta_hidden_layer_weights[i][0] /= len_data
        delta_hidden_layer_weights[i][1] /= len_data
        delta_hidden_layer_bias[i] /= len_data

    for i in range(hidden_layer_neurons):
        delta_output_layer_weights[i] /= len_data

    delta_output_layer_bias /= len_data

    for i in range(hidden_layer_neurons):
        hidden_layer_weights[i][0] -= learning_rate * delta_hidden_layer_weights[i][0]
        hidden_layer_weights[i][1] -= learning_rate * delta_hidden_layer_weights[i][1]
        hidden_layer_biases[i] -= learning_rate * delta_hidden_layer_bias[i]

    for i in range(hidden_layer_neurons):
        output_layer_weights[i] -= learning_rate * delta_output_layer_weights[i]

    output_layer_bias -= learning_rate * delta_output_layer_bias

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: ' \
                f'Age Weights: {[str(hidden_layer_weights[i][0])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'Gender Weights: {[str(hidden_layer_weights[i][1])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'HL Biases: {[str(hidden_layer_biases[i])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'OL Weights: {[str(output_layer_weights[i])[:8] for i in range(hidden_layer_neurons)]} ' \
                f'OL Bias: {output_layer_bias:,.5f}',
        )
        sleep(0.5)

print()

print(f'Final Weights: ' \
        f'Age Weights: {[str(hidden_layer_weights[i][0])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'Gender Weights: {[str(hidden_layer_weights[i][1])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'HL Biases: {[str(hidden_layer_biases[i])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'OL Weights: {[str(output_layer_weights[i])[:8] for i in range(hidden_layer_neurons)]} ' \
        f'OL Bias: {output_layer_bias:,.5f}',
)

def predict(age, gender):
    preactivated_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]
    postactivation_hidden_layer_output_values = [0.0 for _ in range(hidden_layer_neurons)]

    for i in range(hidden_layer_neurons):
        preactivated_hidden_layer_output_values[i] = hidden_layer_weights[i][0] * age + hidden_layer_weights[i][1] * gender + hidden_layer_biases[i]
        postactivation_hidden_layer_output_values[i] = relu(preactivated_hidden_layer_output_values[i])

    z_out = output_layer_bias
    for i in range(hidden_layer_neurons):
        z_out += postactivation_hidden_layer_output_values[i] * output_layer_weights[i]

    y_prediction = sigmoid(z_out)
    return y_prediction

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