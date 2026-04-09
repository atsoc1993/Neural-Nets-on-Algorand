from LinearRegressionClient import PrimeTrainingArgs
from constants import LINEAR_REGRESSION_MODEL_CLIENT, SCALE_FACTOR, LEARNING_RATE, EPOCHS
import logging


'''
-- WARNING --
IF YOU ADJUSTED THE SCALE FACTOR, NOTE THAT YOUR TRAINING DATA HAS ALREADY BEEN SCALED AND STORED
YOU SHOULD DO A FULL RE-RUN w/ __full_run.py
'''
txn_ids = LINEAR_REGRESSION_MODEL_CLIENT.send.prime_training(
    args=PrimeTrainingArgs(
        scale_factor=SCALE_FACTOR,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
).tx_ids

logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")