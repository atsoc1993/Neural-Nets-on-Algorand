from constants import ALGORAND, LINEAR_REGRESSION_APP_ID, LINEAR_REGRESSION_MODEL_CLIENT
from algokit_utils import CommonAppCallParams, AlgoAmount, SendParams
from threading import Thread
import logging
import sys

def send_training_loop_tx(note_index: int, epoch: int):
    try:
        current_round = ALGORAND.client.algod.suggested_params().first
        txn_id = LINEAR_REGRESSION_MODEL_CLIENT.send.run_a_training_loop(
            params=CommonAppCallParams(
                max_fee=AlgoAmount(micro_algo=50_000),
                note=str(note_index).encode(),
                first_valid_round=current_round, # This should be able to be removed in testnet or mainnet, the assumption is block size is smaller by default on localnet
                validity_window=1000 
            ),
            send_params=SendParams(
                cover_app_call_inner_transaction_fees=True
            )
        ).tx_id
        

        logging.info(f"Ran Training Loop for Epoch: {epoch}, Tx ID: {txn_id}")
    except Exception as e:
        logging.error(f"{e}")
        sys.exit()

globals = ALGORAND.app.get_global_state(LINEAR_REGRESSION_APP_ID)
total_epochs = globals.get('epochs').value
epochs_completed = globals.get('epochs_completed').value
epochs = total_epochs - epochs_completed

print(f'Epochs to complete')

note_index = 1
threads: list[Thread] = []
for epoch in range(epochs):
    thread = Thread(target=send_training_loop_tx, args=(note_index, epoch))
    thread.start()
    threads.append(thread)
    note_index += 1

for thread in threads:
    thread.join()

logging.info(f"Completed {epochs} training loops")