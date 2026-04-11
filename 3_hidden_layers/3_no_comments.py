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

epochs = 2_500

hidden_layer_1_input_size = len([ages, genders])

hidden_layer_neurons = 3

hidden_layers = 3
first_hidden_layer = [[[uniform(-.5, .5) for _ in range(hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(1)]

post_hidden_layer_1_input_size = 3
hidden_layer_weights = first_hidden_layer + [[[uniform(-.5, .5) for _ in range(post_hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers - 1)]
hidden_layer_biases = [[uniform(-.5, .5) for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]

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
    delta_first_hidden_layer = [[[0.0 for _ in range(hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(1)]
    delta_hidden_layer_weights = delta_first_hidden_layer + [[[0.0 for _ in range(post_hidden_layer_1_input_size)] for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers - 1)]
    delta_hidden_layer_biases = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
    delta_output_layer_weights = [0.0 for _ in range(hidden_layer_neurons)]
    delta_output_layer_bias = 0.0

    for age, gender, cbp in zip(ages_normalized, genders_normalized, cbps_normalized):
        y = cbp

        features = [age, gender]

        preactivated_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
        postactivation_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]

        for i in range(hidden_layers):
            for j in range(hidden_layer_neurons):
                preactivation_value = hidden_layer_biases[i][j]
                if i == 0:
                    preactivation_value = hidden_layer_biases[i][j]

                    for k in range(hidden_layer_1_input_size):
                        preactivation_value += hidden_layer_weights[i][j][k] * features[k]

                    preactivated_hidden_layer_output_values[i][j] = preactivation_value
                    postactivation_hidden_layer_output_values[i][j] = relu(preactivation_value)

                else:
                    preactivation_value = hidden_layer_biases[i][j]

                    for k in range(post_hidden_layer_1_input_size):
                        preactivation_value += hidden_layer_weights[i][j][k] * postactivation_hidden_layer_output_values[i - 1][k]

                    preactivated_hidden_layer_output_values[i][j] = preactivation_value
                    postactivation_hidden_layer_output_values[i][j] = relu(preactivation_value)

        z_out = output_layer_bias
        for i in range(hidden_layer_neurons):
            z_out += postactivation_hidden_layer_output_values[-1][i] * output_layer_weights[i]

        y_prediction = sigmoid(z_out)

        y_error = y_prediction - cbp

        for i in range(hidden_layer_neurons):
            delta_output_layer_weights[i] += y_error * postactivation_hidden_layer_output_values[-1][i]
        delta_output_layer_bias += y_error

        hidden_layer_errors = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
        for i in range(hidden_layer_neurons):
            hidden_layer_errors[-1][i] = y_error * output_layer_weights[i] * relu_derivative(preactivated_hidden_layer_output_values[-1][i])

        for i in range(hidden_layers - 2, -1, -1):
            for j in range(hidden_layer_neurons):
                weighted_error_sum = 0.0
                for k in range(hidden_layer_neurons):
                    weighted_error_sum += hidden_layer_errors[i + 1][k] * hidden_layer_weights[i + 1][k][j]
                hidden_layer_errors[i][j] = weighted_error_sum * relu_derivative(preactivated_hidden_layer_output_values[i][j])

        for i in range(hidden_layers):
            for j in range(hidden_layer_neurons):
                if i == 0:
                    delta_hidden_layer_weights[i][j][0] += hidden_layer_errors[i][j] * age
                    delta_hidden_layer_weights[i][j][1] += hidden_layer_errors[i][j] * gender
                else:
                    for k in range(post_hidden_layer_1_input_size):
                        delta_hidden_layer_weights[i][j][k] += hidden_layer_errors[i][j] * postactivation_hidden_layer_output_values[i - 1][k]
                delta_hidden_layer_biases[i][j] += hidden_layer_errors[i][j]

    for i in range(hidden_layers):
        for j in range(hidden_layer_neurons):
            for k in range(len(delta_hidden_layer_weights[i][j])):
                delta_hidden_layer_weights[i][j][k] /= len_data
            delta_hidden_layer_biases[i][j] /= len_data

    for i in range(hidden_layer_neurons):
        delta_output_layer_weights[i] /= len_data

    delta_output_layer_bias /= len_data

    for i in range(hidden_layers):
        for j in range(hidden_layer_neurons):
            for k in range(len(hidden_layer_weights[i][j])):
                hidden_layer_weights[i][j][k] -= learning_rate * delta_hidden_layer_weights[i][j][k]
            hidden_layer_biases[i][j] -= learning_rate * delta_hidden_layer_biases[i][j]

    for i in range(hidden_layer_neurons):
        output_layer_weights[i] -= learning_rate * delta_output_layer_weights[i]

    output_layer_bias -= learning_rate * delta_output_layer_bias

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}: ' \
                f'HL Weights: {hidden_layer_weights} ' \
                f'HL Biases: {hidden_layer_biases} ' \
                f'OL Weights: {output_layer_weights} ' \
                f'OL Bias: {output_layer_bias}',
        )
        sleep(0.5)

print()

print(f'Final Weights: ' \
    f'HL Weights: {hidden_layer_weights} ' \
    f'HL Biases: {hidden_layer_biases} ' \
    f'OL Weights: {output_layer_weights} ' \
    f'OL Bias: {output_layer_bias}',
)

def predict(age, gender):
    preactivated_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]
    postactivation_hidden_layer_output_values = [[0.0 for _ in range(hidden_layer_neurons)] for _ in range(hidden_layers)]

    for i in range(hidden_layers):
        for j in range(hidden_layer_neurons):
            if i == 0:
                preactivated_hidden_layer_output_values[i][j] = hidden_layer_weights[i][j][0] * age + hidden_layer_weights[i][j][1] * gender + hidden_layer_biases[i][j]
            else:
                preactivation_value = hidden_layer_biases[i][j]
                for k in range(post_hidden_layer_1_input_size):
                    preactivation_value += hidden_layer_weights[i][j][k] * postactivation_hidden_layer_output_values[i - 1][k]
                preactivated_hidden_layer_output_values[i][j] = preactivation_value

            postactivation_hidden_layer_output_values[i][j] = relu(preactivated_hidden_layer_output_values[i][j])

    z_out = output_layer_bias
    for i in range(hidden_layer_neurons):
        z_out += postactivation_hidden_layer_output_values[-1][i] * output_layer_weights[i]

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