from constants import ALGORAND, LINEAR_REGRESSION_APP_ID, LINEAR_REGRESSION_MODEL_CLIENT
from algokit_utils import CommonAppCallParams, AlgoAmount, SendParams
from threading import Thread, Event, Lock
from time import sleep
from random import randint
import logging


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
        
        txn_id = LINEAR_REGRESSION_MODEL_CLIENT.send.run_training_loops(
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

globals = ALGORAND.app.get_global_state(LINEAR_REGRESSION_APP_ID)
total_epochs = globals.get('epochs').value # type: ignore
epochs_completed = globals.get('epochs_completed').value # type: ignore
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

logging.info(f"Completed {epochs} training loops")