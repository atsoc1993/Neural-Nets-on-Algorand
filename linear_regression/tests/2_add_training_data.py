from constants import ALGORAND, LINEAR_REGRESSION_MODEL_CLIENT, PK, SIGNER, SCALE_FACTOR
from LinearRegressionClient import AddInputsAndTargetsArgs
from algokit_utils import PaymentParams, AlgoAmount, CommonAppCallParams, SendParams
import logging

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
group = LINEAR_REGRESSION_MODEL_CLIENT.new_group()

for x_input_chunk, y_target_chunk in zip(x_input_chunks, y_target_chunks):

    note_index += 1
    
    mbr_payment = ALGORAND.create_transaction.payment(
        PaymentParams(
            sender=PK,
            signer=SIGNER,
            amount=AlgoAmount(micro_algo=100_000),
            receiver=LINEAR_REGRESSION_MODEL_CLIENT.app_address,
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

        group = LINEAR_REGRESSION_MODEL_CLIENT.new_group()
        logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")


txn_ids = group.send(
    send_params=SendParams(
        cover_app_call_inner_transaction_fees=True,
    )
).tx_ids
logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")