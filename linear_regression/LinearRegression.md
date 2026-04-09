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
Decimals — we must use a scale factor on our input and target data pairs, as so that we can maintain integrity of their values, but can still allow them to be interpreted on chain as integers, not floats. This scale factor must be applied off-chain before passing data into the contract. We also apply this scale factor anywhere in our training loop logic where we must multiply our scaled weight/bias with scaled x or y data.

### The Second Obstacle
Negatives — although our equation is positive in slope and y-intercept values (weight & bias), we will still need to handle small negative nudges from our delta weight & delta bias if our predictions go over the expected targets. We account for this with signed add and signed multiply subroutines (functions) in our contract that accept parameters for whether or not weight/bias or delta weight/delta bias or negative, and we also keep a global state for the persistent weight / bias to keep track of their sign (+/-)

# Results

Equation: `y = 19.3x + 72.5`

## On-Chain

Our Globals for Weight and Bias:
```
Weight: 1930002473966
Bias: 7249949733874
```

Scaled Down:
```
Weight: 19.30002473966
Bias: 72.49949733874
```

Prediction for f(6) = 188.3:
```188.2996457767```

## Off-chain linear regression model tests:
Weight and Bias:
```
Weight: 19.300024739445377
Bias: 72.4994973430938
```

Prediction for f(6) = 188.3:
```
188.29964577976605
```