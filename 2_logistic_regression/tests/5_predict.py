from LogisticRegressionModelClient import PredictArgs
from constants import LOGISTIC_REGRESSION_CLIENT, LOGISTIC_REGRESSION_APP_ID, ALGORAND, SCALE_FACTOR

globals_ = ALGORAND.app.get_global_state(LOGISTIC_REGRESSION_APP_ID)

age_weight = globals_.get('age_weight_magnitude').value / SCALE_FACTOR # type: ignore
age_weight_is_negative = globals_.get('age_weight_is_negative').value # type: ignore

gender_weight = globals_.get('gender_weight_magnitude').value / SCALE_FACTOR # type: ignore
gender_weight_is_negative = globals_.get('gender_weight_is_negative').value # type: ignore

bias = globals_.get('bias_magnitude').value / SCALE_FACTOR # type: ignore
bias_is_negative = globals_.get('bias_is_negative').value # type: ignore

if age_weight_is_negative:
    age_weight *= -1

if gender_weight_is_negative:
    gender_weight *= -1

if bias_is_negative:
    bias *= -1

print(f'Age Weight: {age_weight}')
print(f'Gender Weight: {gender_weight}')
print(f'Bias: {bias}')

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

    txn_response = LOGISTIC_REGRESSION_CLIENT.send.predict(
        args=PredictArgs(
            age=int(normalized_age * SCALE_FACTOR),
            gender=int(normalized_gender * SCALE_FACTOR)
        )
    )

    prediction_scaled = txn_response.abi_return
    prediction = prediction_scaled / SCALE_FACTOR # type: ignore

    cbp_predicted_y = 'Yes' if prediction > 0.5 else 'No'
    expected_y = 'Yes' if gender_str == 'Female' and age_int < 50 else 'No'

    print(f'Prediction for {age_int} year old {gender_str}: {cbp_predicted_y} ({prediction}); Expected: {expected_y}')