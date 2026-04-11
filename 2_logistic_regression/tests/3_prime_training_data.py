from LogisticRegressionModelClient import PrimeTrainingArgs
from constants import LOGISTIC_REGRESSION_CLIENT, SCALE_FACTOR, LEARNING_RATE, EPOCHS
import logging

txn_ids = LOGISTIC_REGRESSION_CLIENT.send.prime_training(
    args=PrimeTrainingArgs(
        scale_factor=SCALE_FACTOR,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS
    )
).tx_ids

logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")