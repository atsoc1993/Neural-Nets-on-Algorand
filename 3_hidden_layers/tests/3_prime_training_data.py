from LogisticNeuralNetworkClient import PrimeTrainingArgs
from algokit_utils import CommonAppCallParams, AlgoAmount, PaymentParams
from constants import (
    ALGORAND,
    PK,
    SIGNER,
    LOGISTIC_NEURAL_NETWORK_CLIENT,
    SCALE_FACTOR,
    LEARNING_RATE,
    EPOCHS,
    HIDDEN_LAYERS,
    HIDDEN_NEURONS,
)
import logging

mbr_payment = ALGORAND.create_transaction.payment(
    PaymentParams(
        sender=PK,
        signer=SIGNER,
        amount=AlgoAmount(micro_algo=5_000_000),
        receiver=LOGISTIC_NEURAL_NETWORK_CLIENT.app_address,
        validity_window=1000,
    )
)

txn_ids = LOGISTIC_NEURAL_NETWORK_CLIENT.send.prime_training(
    args=PrimeTrainingArgs(
        scale_factor=SCALE_FACTOR,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        hidden_layers=HIDDEN_LAYERS,
        hidden_neurons=HIDDEN_NEURONS,
        mbr_payment=mbr_payment
    ),
    params=CommonAppCallParams(
        max_fee=AlgoAmount(micro_algo=255_000)
    ),
    send_params={
        'cover_app_call_inner_transaction_fees': True,
        'populate_app_call_resources': True
    }
).tx_ids

logging.info(f"{len(txn_ids)} Transactions Submitted: {txn_ids}")