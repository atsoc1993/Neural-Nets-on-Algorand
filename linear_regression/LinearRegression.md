~~algokit compile py linear_regression_contract.py --out-dir contract_files --output-arc56~~
### Generate Contract Files
_Note: Until a fix is in place for downstream dependency, use `-O 0` flag & param to lower optimization level:_

`algokit compile py linear_regression_contract.py --out-dir contract_files --output-arc56 -O 0`

### Generate Client
Run the following command within the linear_regression folder
`algokitgen-py -a 'contract_files/LinearRegressionModel.arc56.json' -o 'tests/LinearRegressionClient.py'`