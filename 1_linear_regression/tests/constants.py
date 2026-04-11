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

LINEAR_REGRESSION_APP_ID = int(os.getenv('LINEAR_REGRESSION_APP_ID', 0))
if LINEAR_REGRESSION_APP_ID != 0:
    LINEAR_REGRESSION_MODEL_CLIENT = ALGORAND.client.get_typed_app_client_by_id(
        app_id=LINEAR_REGRESSION_APP_ID,
        default_sender=PK,
        default_signer=SIGNER,
        approval_source_map=Program(program=APP_SPEC.source.get_decoded_approval(), client=ALGORAND.client.algod).source_map, # type: ignore
        typed_client=LinearRegressionModelClient
    )
else:
    logging.info("Warning: Linear Regression APP ID does not exist yet and client cannot be created, please run 1_deploy_contract.py if you are not doing so now")

# AVOID ADJUSTING SCALE_FACTOR, IT IS MAINLY FOR HIGHER PRECISION, 
# LOWER SCALE_FACTORS MAY RESULT IN LESS ACCURATE RESULTS
# EXTREMELY HIGH SCALE FACTORS MAY RESULT IN OVERFLOW ERRORS ("MATH ATTEMPTED ON LARGE BYTE-ARRAY")
SCALE_FACTOR = 100_000_000_000

# AVOID HIGH LEARNING RATES, THESE WILL RESULT IN "MATH ATTEMPTED ON LARGE BYTE-ARRAY WHEN PAIRED WITH LARGE SCALE FACTORS"
LEARNING_RATE = int(0.005 * SCALE_FACTOR)
EPOCHS = 10_000
