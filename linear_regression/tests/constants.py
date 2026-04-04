from LinearRegressionClient import LinearRegressionModelFactory, LinearRegressionModelClient, APP_SPEC
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

LINEAR_REGRESSION_FACTORY = LinearRegressionModelFactory(
    algorand=ALGORAND,
    default_sender=PK,
    default_signer=SIGNER,
)

LINEAR_REGRESSION_APP_ID = os.getenv('LINEAR_REGRESSION_APP_ID', '')
if LINEAR_REGRESSION_APP_ID != '':
    LINEAR_REGRESSION_MODEL_CLIENT = ALGORAND.client.get_typed_app_client_by_id(
        app_id=int(LINEAR_REGRESSION_APP_ID),
        default_sender=PK,
        default_signer=SIGNER,
        approval_source_map=Program(program=APP_SPEC.source.get_decoded_approval(), client=ALGORAND.client.algod).source_map,
        typed_client=LinearRegressionModelClient
    )
else:
    logging.info("Warning: Linear Regression APP ID does not exist yet and client cannot be created, please run 1_deploy_contract.py if you are not doing so now")