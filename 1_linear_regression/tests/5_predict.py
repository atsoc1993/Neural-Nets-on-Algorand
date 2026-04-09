from LinearRegressionClient import PredictArgs
from algosdk.abi import ABIType
from constants import LINEAR_REGRESSION_MODEL_CLIENT, LINEAR_REGRESSION_APP_ID, ALGORAND, SCALE_FACTOR

globals = ALGORAND.app.get_global_state(LINEAR_REGRESSION_APP_ID)
weight = globals.get('weight_magnitude').value / SCALE_FACTOR
bias = globals.get('bias_magnitude').value / SCALE_FACTOR

'''
# y = 19.3x + 72.5
'''
print(f'Weight: {weight}')
print(f'Bias: {bias}')

txn_response = LINEAR_REGRESSION_MODEL_CLIENT.send.predict(
    args=PredictArgs(
        x=6 * SCALE_FACTOR
    )
)

txn_result = txn_response.abi_return
prediction = (txn_result[0] * -1 if txn_result[1] else txn_result[0]) / SCALE_FACTOR

print(f'Prediction: {prediction}')


