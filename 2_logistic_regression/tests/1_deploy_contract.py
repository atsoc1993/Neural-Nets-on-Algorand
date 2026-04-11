from algokit_utils import PaymentParams, AlgoAmount
from dotenv import set_key
from constants import ALGORAND, PK, SIGNER, LOGISTIC_REGRESSION_FACTORY
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