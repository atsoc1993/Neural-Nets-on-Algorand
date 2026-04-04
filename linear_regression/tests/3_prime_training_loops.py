from LinearRegressionClient import PrimeTrainingArgs
from constants import LINEAR_REGRESSION_MODEL_CLIENT, SCALE_FACTOR
import logging

learning_rate = int(0.001 * SCALE_FACTOR)
epochs = 10_000

txn_ids = LINEAR_REGRESSION_MODEL_CLIENT.send.prime_training(
    args=PrimeTrainingArgs(
        learning_rate=learning_rate,
        epochs=epochs
    )
).tx_ids

logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")