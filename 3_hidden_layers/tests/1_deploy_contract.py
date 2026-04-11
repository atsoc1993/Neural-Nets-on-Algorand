from algokit_utils import PaymentParams, AlgoAmount
from dotenv import set_key
from constants import ALGORAND, PK, SIGNER, LOGISTIC_NEURAL_NETWORK_FACTORY
import logging

logging.info('Deploying Logistic Neural Network App . . .')

logistic_neural_network_client, txn_result = LOGISTIC_NEURAL_NETWORK_FACTORY.send.create.bare()
logging.info(f'Deployed Logistic Neural Network App, App ID: {logistic_neural_network_client.app_id}')

set_key('.env', 'LOGISTIC_NEURAL_NETWORK_APP_ID', str(logistic_neural_network_client.app_id))
logging.info('Saved App ID to .env under key: LOGISTIC_NEURAL_NETWORK_APP_ID')

logging.info('Funding Account MBR to Logistic Neural Network App . . .')

ALGORAND.send.payment(
    params=PaymentParams(
        sender=PK,
        signer=SIGNER,
        amount=AlgoAmount(micro_algo=100_000),
        receiver=logistic_neural_network_client.app_address,
        validity_window=1000
    )
)

logging.info('Funded Account MBR to Logistic Neural Network App')