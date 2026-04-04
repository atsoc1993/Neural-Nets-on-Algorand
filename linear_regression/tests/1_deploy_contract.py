from algokit_utils import PaymentParams, AlgoAmount
from dotenv import set_key
from constants import ALGORAND, PK, SIGNER, LINEAR_REGRESSION_FACTORY

print(f'Deploying Linear Regression Model App . . .')
linear_regression_client, txn_result = LINEAR_REGRESSION_FACTORY.send.create.bare()
print(f'Deployed Linear Regression Model App, App ID: {linear_regression_client.app_id}')
set_key('.env', 'LINEAR_REGRESSION_APP_ID', str(linear_regression_client.app_id))
print(f'Saved App ID to .env under key: LINEAR_REGRESSION_APP_ID')

print(f'Funding Account MBR to Linear Regression Model App . . .')
fund_linear_regression_app_tx = ALGORAND.send.payment(
    params=PaymentParams(
        sender=PK,
        signer=SIGNER,
        amount=AlgoAmount(micro_algo=100_000),
        receiver=linear_regression_client.app_address,
        validity_window=1000
    )
)
print(f'Funded Account MBR to Linear Regression Model App')




