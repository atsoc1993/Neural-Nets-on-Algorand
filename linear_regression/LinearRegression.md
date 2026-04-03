~~algokit compile py linear_regression_contract.py --out-dir contract_files --output-arc56~~
### Generate Contract Files
_Note: Until a fix is in place for downstream dependency, use `-O 0` flag & param to lower optimization level:_

`algokit compile py linear_regression_contract.py --out-dir contract_files --output-arc56 -O 0`

### Generate Client
`algokitgen-py -a 'contract_files/LinearRegressionModel.arc56.json' -o 'LinearRegressionClient.py'`