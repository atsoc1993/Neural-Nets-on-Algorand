from algokit_utils import PaymentParams, AlgoAmount, CommonAppCallParams, SendParams
from LogisticRegressionModelClient import PrimeTrainingArgs, AddInputsAndTargetsArgs, PredictArgs
from dotenv import set_key
from constants import ALGORAND, PK, SIGNER, LOGISTIC_REGRESSION_FACTORY, SCALE_FACTOR, LEARNING_RATE, EPOCHS
from threading import Thread, Event, Lock
from time import sleep
from random import randint
import logging

logging.info('Deploying Logistic Regression Model App . . .')

logistic_regression_client, txn_result = LOGISTIC_REGRESSION_FACTORY.send.create.bare()
logging.info(f'Deployed Logistic Regression Model App, App ID: {logistic_regression_client.app_id}')

set_key('.env', 'LOGISTIC_REGRESSION_APP_ID', str(logistic_regression_client.app_id))
logging.info('Saved App ID to .env under key: LOGISTIC_REGRESSION_APP_ID')

logging.info('Funding Account MBR to Logistic Regression Model App . . .')

ALGORAND.send.payment(
    params=PaymentParams(
        sender=PK,
        signer=SIGNER,
        amount=AlgoAmount(micro_algo=100_000),
        receiver=logistic_regression_client.app_address,
        validity_window=1000
    )
)
logging.info('Funded Account MBR to Logistic Regression Model App')

ages = [i for i in range(20, 50)] * 2 + [i for i in range(50, 80)] * 2
genders = (['Female'] * 30 + ['Male'] * 30) * 2
cbps = ['Yes'] * 30 + ['No'] * 90

genders_normalized = [0 if gender == 'Male' else 1 for gender in genders]
cbps_normalized = [0 if cbp == 'No' else 1 for cbp in cbps]

def normalize_age(age: int) -> float:
    return (age - 20) / 60

ages_normalized = [normalize_age(age) for age in ages]

ages_scaled = [int(age * SCALE_FACTOR) for age in ages_normalized]
genders_scaled = [int(gender * SCALE_FACTOR) for gender in genders_normalized]
targets_scaled = [int(target * SCALE_FACTOR) for target in cbps_normalized]

chunk_size = 10
age_chunks = [ages_scaled[i:i + chunk_size] for i in range(0, len(ages_scaled), chunk_size)]
gender_chunks = [genders_scaled[i:i + chunk_size] for i in range(0, len(genders_scaled), chunk_size)]
target_chunks = [targets_scaled[i:i + chunk_size] for i in range(0, len(targets_scaled), chunk_size)]

max_group_size = 16
note_index = 1
group_size = 0
group = logistic_regression_client.new_group()

for age_chunk, gender_chunk, target_chunk in zip(age_chunks, gender_chunks, target_chunks):
    note_index += 1

    mbr_payment = ALGORAND.create_transaction.payment(
        PaymentParams(
            sender=PK,
            signer=SIGNER,
            amount=AlgoAmount(micro_algo=300_000),
            receiver=logistic_regression_client.app_address,
            validity_window=1000,
            note=str(note_index).encode()
        )
    )

    group.add_inputs_and_targets(
        args=AddInputsAndTargetsArgs(
            parts_age_inputs=age_chunk,
            parts_gender_inputs=gender_chunk,
            parts_y_targets=target_chunk,
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
        group = logistic_regression_client.new_group()
        group_size = 0
        logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")

if group_size > 0:
    txn_ids = group.send(
        send_params=SendParams(
            cover_app_call_inner_transaction_fees=True,
        )
    ).tx_ids
    logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")

txn_ids = logistic_regression_client.send.prime_training(
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
        for _ in range(sleep_amount):
            sleep(1)
            if stop_event.is_set():
                return

        if stop_event.is_set():
            return

        txn_id = logistic_regression_client.send.run_training_loops(
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

globals_ = ALGORAND.app.get_global_state(logistic_regression_client.app_id)
total_epochs = globals_.get('epochs').value # type: ignore
epochs_completed = globals_.get('epochs_completed').value # type: ignore
epochs = total_epochs - epochs_completed # type: ignore

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

globals_ = ALGORAND.app.get_global_state(logistic_regression_client.app_id)

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

for age_int, gender_str in tests:
    normalized_age = normalize_age(age_int)
    normalized_gender = 0 if gender_str == 'Male' else 1

    txn_response = logistic_regression_client.send.predict(
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