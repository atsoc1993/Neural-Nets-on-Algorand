~~algokit compile py linear_regression_contract.py --out-dir contract_files --output-arc56~~
### Generate Contract Files
_Note: Until a fix is in place for downstream dependency, use `-O 0` flag & param to lower optimization level:_

`algokit compile py linear_regression_contract.py --out-dir contract_files --output-arc56 -O 0`

### Generate Client
Run the following command within the linear_regression folder
`algokitgen-py -a 'contract_files/LinearRegressionModel.arc56.json' -o 'tests/LinearRegressionClient.py'`


### Viewing Transactions
Use algokit localnet explore

# On Chain Implementation

### The First Obstacle
Decimals — we must use a scale factor on our input and target data pairs so that we can maintain integrity of their values, but can still allow them to be interpreted on chain as integers, not floats. This scale factor must be applied off-chain before passing data into the contract.


### The Second Obstacle
Negatives — 

