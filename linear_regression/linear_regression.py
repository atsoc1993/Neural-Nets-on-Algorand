# Linear Regression, singular perceptron
''' 
    Linear Regression Models are the simplest possible form is a singular neuron in a neural network, and the value of x is 10, y = 27
    The majority of the logic behind what makes them work fits into an equation we are all very familiar with:
    
    ==> The Equation of a Line <==

    In the equation y = mx + b:
    - m, the slope, is the equivalent of a weight in a neural network
    - b, the y-intercept, is the equivalent of a bias in a neural network
    - x & y, are of course, the inputs and targets. 

    Where x is some number, what is the value of y?

    Take the equation: y = 2x + 7
    If x was say... 10, what would y be?
    Simply plug in x and solve for y:

    f(x) = y

    y = 2x + 7
    y = 2(10) + 7
    y = 20 + 7
    y = 27

    f(20) = 27

    So when the f(x) (function of x) is 2x + 7, and the value of x is 10, y is 27

    NOTE that a linear regression model works specifically with CONTINUOUS data, meaning quantifiable values, unlike a classification-esque approach with DISCRETE data:
    eg; "cat", "bird", "dog" vs. [200, 400, 800, 1600]

    In a linear regression model, the idea is we are basically training a singular neuron (perceptron)
      to DISCOVER the equation given a series of related X and Y values.

    With the equation of a line in mind: y = mx + b
    Basically we start off with y = 0x + 0, where m and b start off at ZERO.
    We then feed the perceptron some input and target hundreds, if not thousands of times (training loops),
    and it nudges the weight (m [slope]) and the bias (b [y-intercept])


    Don't worry about this for now but in practice these values are actually not usually 0 — they usually are slightly negative or slightly positive
    Which we can achieve with random.uniform(-.5, .5) — but for a Linear Regression model, we can safely just use zero.
    
'''
# So we can observe training loop updates, import the time module for sleeping later
import time 

# Define some base x values, these are arbitrary for linear regression models assuming we don't use exponents
x_inputs = [i for i in range(1, 31)] # [1, 2, ... 30]

# Example equation of a line we will use (y = mx + b):
# y = 19.3x + 72.5

# Create our y values from the above equation f(x) for each x input we created
y_targets = [19.3 * x + 72.5 for x in x_inputs] # [91.8, 111.1, ... 651.5]

# Define our initial weight and bias:
weight = 0.0
bias = 0.0

# Define the amount of training loops (epochs) and the learning rate
# Note: These fields require tinkering at times to avoid underfitting or overfitting, 
# but straight forward for linear regression of a non-parabolic (straight line, doesnt curve upwards or downwards)
# Recommended: 
# More epochs, Lower learning rate — 
# Less Epochs, Higher Learning Rate
epochs = 10_000
learning_rate = 0.001

# Begin our training loop
for epoch in range(epochs):

    # Keep track of "deltas" in each training loop, these are reset each time to 0
    # These "differences" or "total errors" are what we use to update our global weight and bias at lines 54 & 55
    # We can also keep track of the total error between training loops, which you'll see how we calculate shortly
    delta_weight = 0.0
    delta_bias = 0.0
    total_error = 0.0

    # Iterate through each zipped x & y inputs and targets (outputs):
    for x, y in zip(x_inputs, y_targets): # [(1, 91.8), (2, 111.1), ... (30, 651.5)]
        
        # Our "Guess" of Y is basically trying to use the "x" value we are at now to solve the equation
        # where m and b are our global weight and bias at 54 & 55
        # these are 0 at first, but gradually inch towards the accurate values, which are 19.3 for m, and 72.5 for b 
        # We defined the equation earlier on line 51 in list comprehension
        guessed_y = weight * x + bias

        # We have a guess for y, now what? Let's see how wrong we were:
        y_error = guessed_y - y 
        # ^^^ Where x is 1, in our very first training loop (epoch), y should be 91.8 — 
        # this means y_error is also 91.8 here, since m is 0 and b is 0 initially: y = 0 * 1 + 0 = 0
        # Incredibly wrong.

        # Lets tell our *DELTA weight and *DELTA bias they need to tell our global weight and global bias to step up their game, 
        # and how poorly they're doing right now at our first iteration

        # When calculating the error for a weight delta, we always multiply the error by the input value used for that weight
        # The bias delta can accept the y_error as is without any other arithmetic
        delta_weight += y_error * x
        delta_bias += y_error
        total_error += y_error

    
    # The end of a single training loop (epoch), many more to go
    # Before we apply the changes to our global weight and bias we need
    # to average them based on the number of inputs and targets we used
    # Just for visibility we will define the length of inputs here, 
    # but optimally you would declare them outside this loop to be more efficient
    len_inputs = len(x_inputs)

    delta_weight /= len_inputs
    delta_bias /= len_inputs

    # Quick stop for additional notes (These can be skipped)

    # Additional Notes:
    # Optionally, you can assert that the inputs and targets (y values) are the same length
    # assert len(set(len(x) for x in (x_inputs, y_targets))) == 1, "Input & Output Data Length Mismatch"

    # There are ways to handle missing, or abnormal data (anomalies or NaN values), or should they not be the same length — but we 
    # will not see examples of this throughout this repo. 
    # But know that what usually happens (although there are different options) is extreme values or "missing" values in a data set are normalized by being replaced 
    # with MODES or MEANS of the data set for that category, depending on if the values are discrete or continuous.
    # A Mode is just the most commonly occuring value in a set of data, eg: [0, 1, 1, 2], the mode is 1.

    # Continuing back into our linear regression model training loop. . .

    # We can now apply these *delta weight & bias to our *global weight & bias, and make sure to use our learning rate when applying them
    # We "apply" them by substracting the product of the deltas with the learning rate from the global weight and bias
    weight -= delta_weight * learning_rate
    bias -= delta_bias * learning_rate

    # Optionally print weight, bias, and errors every 100 or so epochs
    # The total errors should decrease overtime (Become more positive)
    # Technical Note: The way this print statement is written, it will not create new lines, and instead rewrite in place — which looks super cool.
    # Note that it doesn't rewrite onto the same line, on my PC at least, if the print exceeds the length of 1 line — 
    # you can use CTRL or COMMAND + or - to zoom in and out to prevent this, or format the number of decimal points shown in the string literal
    # Clear your terminal with CTRL or COMMAND and L key — or type "clear" into the console and press enter
    if epoch % 100 == 0:
        print(f'\rEpoch: {epoch}; Weight: {weight:,.5f}; Bias: {bias:,.5f}; Total Error: {total_error:,.5f}', end='')
        # Depending on your device, this code may run exceptionally fast and you won't be able to observe the changes,
        # so for observability we add this line below, but you can remove the line below if you feel it is hindering you
        time.sleep(0.1)

# Clear out terminal return to the next line for forthcoming print statement
print('\n')

# Once all the training loops have finished, you can now use your perceptron's trained weight and bias as you'd like
# Here's all the values of x_inputs and y_inputs to test against

# print([[x, y] for x, y in zip(x_inputs, y_targets)])
'''
[
    [1, 91.8],   [2, 111.1],  [3, 130.4],  [4, 149.7],  [5, 169.0], 
    [6, 188.3],  [7, 207.6],  [8, 226.9],  [9, 246.2],  [10, 265.5], 
    [11, 284.8], [12, 304.1], [13, 323.4], [14, 342.7], [15, 362.0],
    [16, 381.3], [17, 400.6], [18, 419.9], [19, 439.2], [20, 458.5], 
    [21, 477.8], [22, 497.1], [23, 516.4], [24, 535.7], [25, 555.0], 
    [26, 574.3], [27, 593.6], [28, 612.9], [29, 632.2], [30, 651.5]
]        
'''

test_x = 6
predicted_y = weight * test_x + bias
expected_y = 19.3 * test_x + 72.5 # 188.3
print(f'Predicted for f({test_x}): {predicted_y}; Correct Y: {expected_y}; Error: {predicted_y - expected_y}')






