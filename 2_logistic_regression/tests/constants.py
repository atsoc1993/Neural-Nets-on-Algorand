from LogisticRegressionModelClient import LogisticRegressionModelFactory, LogisticRegressionModelClient, APP_SPEC
from algokit_utils import AlgorandClient, SigningAccount, Program
from dotenv import load_dotenv
from algosdk.mnemonic import to_private_key
from algosdk.account import address_from_private_key
import logging
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)

MNEMONIC = os.getenv('MNEMONIC', '')
assert MNEMONIC != '', "Mnemonic not loading from .env, create a 'MNEMONIC' secret in .env from localnet instructions in README.md"
SK = to_private_key(MNEMONIC)
PK = address_from_private_key(SK)
SIGNER = SigningAccount(private_key=SK, address=PK).signer

ALGORAND = AlgorandClient.default_localnet()

LOGISTIC_REGRESSION_FACTORY = LogisticRegressionModelFactory(
    algorand=ALGORAND,
    default_sender=PK,
    default_signer=SIGNER,
)

LOGISTIC_REGRESSION_APP_ID = int(os.getenv('LOGISTIC_REGRESSION_APP_ID', 0))
if LOGISTIC_REGRESSION_APP_ID != 0:
    LOGISTIC_REGRESSION_CLIENT = ALGORAND.client.get_typed_app_client_by_id(
        app_id=LOGISTIC_REGRESSION_APP_ID,
        default_sender=PK,
        default_signer=SIGNER,
        approval_source_map=Program(
            program=APP_SPEC.source.get_decoded_approval(), # type: ignore
            client=ALGORAND.client.algod
        ).source_map,
        typed_client=LogisticRegressionModelClient
    )
else:
    logging.info("Warning: Logistic Regression APP ID does not exist yet and client cannot be created, please run 1_deploy_contract.py if you are not doing so now")

SCALE_FACTOR = 100_000_000_000
LEARNING_RATE = int(0.1 * SCALE_FACTOR)
EPOCHS = 10_000