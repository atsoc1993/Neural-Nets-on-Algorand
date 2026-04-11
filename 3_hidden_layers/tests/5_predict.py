from LogisticNeuralNetworkClient import PredictArgs
from constants import LOGISTIC_NEURAL_NETWORK_CLIENT, LOGISTIC_NEURAL_NETWORK_APP_ID, ALGORAND, SCALE_FACTOR

globals_ = ALGORAND.app.get_global_state(LOGISTIC_NEURAL_NETWORK_APP_ID)

hidden_layers = globals_.get('hidden_layers').value # type: ignore
hidden_neurons = globals_.get('hidden_neurons').value # type: ignore

print(f'Hidden Layers: {hidden_layers}')
print(f'Hidden Neurons: {hidden_neurons}')

tests: list[tuple[int, str]] = [
    (20, 'Female'),
    (80, 'Female'),
    (20, 'Male'),
    (80, 'Male')
]

def normalize_age(age: int) -> float:
    return (age - 20) / 60

for age_int, gender_str in tests:
    normalized_age = normalize_age(age_int)
    normalized_gender = 0 if gender_str == 'Male' else 1

    txn_response = LOGISTIC_NEURAL_NETWORK_CLIENT.send.predict(
        args=PredictArgs(
            age=int(normalized_age * SCALE_FACTOR),
            gender=int(normalized_gender * SCALE_FACTOR)
        )
    )

    prediction_scaled = txn_response.abi_return
    prediction = prediction_scaled / SCALE_FACTOR

    cbp_predicted_y = 'Yes' if prediction > 0.5 else 'No'
    expected_y = 'Yes' if gender_str == 'Female' and age_int < 50 else 'No'

    print(f'Prediction for {age_int} year old {gender_str}: {cbp_predicted_y} ({prediction}); Expected: {expected_y}')