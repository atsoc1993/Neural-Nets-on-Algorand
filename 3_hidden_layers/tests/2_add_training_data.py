from constants import ALGORAND, LOGISTIC_NEURAL_NETWORK_CLIENT, PK, SIGNER, SCALE_FACTOR
from LogisticNeuralNetworkClient import AddInputsAndTargetsArgs
from algokit_utils import PaymentParams, AlgoAmount, CommonAppCallParams, SendParams
import logging

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
group = LOGISTIC_NEURAL_NETWORK_CLIENT.new_group()

for age_chunk, gender_chunk, target_chunk in zip(age_chunks, gender_chunks, target_chunks):
    note_index += 1

    mbr_payment = ALGORAND.create_transaction.payment(
        PaymentParams(
            sender=PK,
            signer=SIGNER,
            amount=AlgoAmount(micro_algo=100_000),
            receiver=LOGISTIC_NEURAL_NETWORK_CLIENT.app_address,
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

        logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")
        group = LOGISTIC_NEURAL_NETWORK_CLIENT.new_group()
        group_size = 0

if group_size > 0:
    txn_ids = group.send(
        send_params=SendParams(
            cover_app_call_inner_transaction_fees=True,
        )
    ).tx_ids
    logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")