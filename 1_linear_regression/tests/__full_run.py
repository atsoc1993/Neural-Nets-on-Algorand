from algokit_utils import PaymentParams, AlgoAmount, CommonAppCallParams, SendParams
from LinearRegressionClient import PrimeTrainingArgs, AddInputsAndTargetsArgs, PredictArgs
from dotenv import set_key
from constants import ALGORAND, PK, SIGNER, LINEAR_REGRESSION_FACTORY, SCALE_FACTOR, LEARNING_RATE, EPOCHS
from threading import Thread, Event, Lock
from time import sleep
from random import randint
import logging
from constants import ALGORAND, SCALE_FACTOR

logging.info(f'Deploying Linear Regression Model App . . .')

# Deploy an instance of the app on localnet
linear_regression_client, txn_result = LINEAR_REGRESSION_FACTORY.send.create.bare()
logging.info(f'Deployed Linear Regression Model App, App ID: {linear_regression_client.app_id}')

# Write app ID to .env
set_key('.env', 'LINEAR_REGRESSION_APP_ID', str(linear_regression_client.app_id))
logging.info(f'Saved App ID to .env under key: LINEAR_REGRESSION_APP_ID')

logging.info(f'Funding Account MBR to Linear Regression Model App . . .')

# Fund 0.1 Algo Account minimum balance requirement to app
fund_linear_regression_app_tx = ALGORAND.send.payment(
    params=PaymentParams(
        sender=PK,
        signer=SIGNER,
        amount=AlgoAmount(micro_algo=100_000),
        receiver=linear_regression_client.app_address,
        validity_window=1000
    )
)
logging.info(f'Funded Account MBR to Linear Regression Model App')



# Define some base x values, these are arbitrary for linear regression models assuming we don't use exponents
x_inputs = [i for i in range(1, 31)] # [1, 2, ... 30]

# Example equation of a line we will use (y = mx + b):
# y = 19.3x + 72.5

# Create our y values from the above equation f(x) for each x input we created
y_targets = [19.3 * x + 72.5 for x in x_inputs] # [91.8, 111.1, ... 651.5]

x_inputs_scaled = [int(x * SCALE_FACTOR) for x in x_inputs]
y_targets_scaled = [int(y * SCALE_FACTOR) for y in y_targets]

# To avoid exceeding collective application argument length, chunk the inputs and targets:
chunk_size = 10
x_input_chunks = [x_inputs_scaled[i:i + 10] for i in range(0, len(x_inputs_scaled), chunk_size)]
y_target_chunks = [y_targets_scaled[i:i + 10] for i in range(0, len(y_targets_scaled), chunk_size)]


transaction_ids = []

max_group_size = 16
note_index = 1

group_size = 0
group = linear_regression_client.new_group()

for x_input_chunk, y_target_chunk in zip(x_input_chunks, y_target_chunks):

    note_index += 1
    
    mbr_payment = ALGORAND.create_transaction.payment(
        PaymentParams(
            sender=PK,
            signer=SIGNER,
            amount=AlgoAmount(micro_algo=100_000),
            receiver=linear_regression_client.app_address,
            validity_window=1000,
            note=str(note_index).encode() # Increment note index by 1 to avoid duplicate txns
        )
    )

    group.add_inputs_and_targets(
        args=AddInputsAndTargetsArgs(
            parts_x_inputs=x_input_chunk,
            parts_y_targets=y_target_chunk,
            mbr_payment=mbr_payment
        ),
        params=CommonAppCallParams(
            max_fee=AlgoAmount(micro_algo=5000)
        ),
    )

    group_size += 2

    if group_size == max_group_size:
        
        
        txn_ids = group.send(
            send_params=SendParams(
                cover_app_call_inner_transaction_fees=True,
            )
        ).tx_ids

        group = linear_regression_client.new_group()
        logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")


txn_ids = group.send(
    send_params=SendParams(
        cover_app_call_inner_transaction_fees=True,
    )
).tx_ids
logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")


txn_ids = linear_regression_client.send.prime_training(
    args=PrimeTrainingArgs(
        scale_factor=SCALE_FACTOR,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
).tx_ids

logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")

stop_event = Event()
error_lock = Lock()

def send_run_training_loops_tx(note_index: int, epoch: int):
    if stop_event.is_set():
        return
    
    try:
        sleep_amount = randint(1, 400)
        for i in range(sleep_amount):
            sleep(1)
            if stop_event.is_set():
                return
            
        if stop_event.is_set():
            return
        
        txn_id = linear_regression_client.send.run_training_loops(
            params=CommonAppCallParams(
                max_fee=AlgoAmount(micro_algo=1_000_000),
                note=str(note_index).encode(),
                validity_window=1000 
            ),
            send_params=SendParams(
                cover_app_call_inner_transaction_fees=True
            )
        ).tx_id
        
        logging.info(f"Ran several Training Loops ... , Tx ID: {txn_id}")
    except Exception as e:
        with error_lock:
            if not stop_event.is_set():
                logging.error(f"Stopping all threads due to error on epoch {epoch}: {e}")
                stop_event.set()

globals = ALGORAND.app.get_global_state(linear_regression_client.app_id)
total_epochs = globals.get('epochs').value
epochs_completed = globals.get('epochs_completed').value
epochs = total_epochs - epochs_completed

print(f'Epochs to complete: {epochs}')

note_index = 1
threads: list[Thread] = []
for epoch in range(epochs):
    thread = Thread(target=send_run_training_loops_tx, args=(note_index, epoch))
    thread.start()
    threads.append(thread)
    note_index += 1

for thread in threads:
    thread.join()

logging.info(f"Completed all {epochs}")

globals = ALGORAND.app.get_global_state(linear_regression_client.app_id)
weight = globals.get('weight_magnitude').value / SCALE_FACTOR
bias = globals.get('bias_magnitude').value / SCALE_FACTOR

'''
# y = 19.3x + 72.5
'''
print(f'Weight: {weight}')
print(f'Bias: {bias}')

txn_response = linear_regression_client.send.predict(
    args=PredictArgs(
        x=2 * SCALE_FACTOR
    )
)

txn_result = txn_response.abi_return
prediction = (txn_result[0] * -1 if txn_result[1] else txn_result[0]) / SCALE_FACTOR

print(f'Prediction: {prediction}')
