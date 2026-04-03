#TODO Convert Linear Regression Model to Contract State, then create training files and test contract predictions
from algopy import ARC4Contract, UInt64, arc4, Box, gtxn, Global, Txn, itxn, urange
from algopy.arc4 import abimethod, DynamicArray

Data = DynamicArray[arc4.UInt64]

class LinearRegressionModel(ARC4Contract):
    def __init__(self) -> None:
        self.weight = UInt64(0)
        self.bias = UInt64(0)
        self.epochs = UInt64(0)
        self.epochs_completed = UInt64(0)
        self.learning_rate = UInt64(0)
        self.x_inputs_box = Box(DynamicArray[arc4.UInt64], key='inputs')
        self.y_targets_box = Box(DynamicArray[arc4.UInt64], key='targets')
        self.length_data = UInt64(0)
        self.ready_for_training = False
        

    @abimethod
    def prime_training(
        self,
        learning_rate: UInt64,
        epochs: UInt64
    ) -> None:
        '''Reset Weight & Bias and set the intended epochs and learning rate for new training phase'''

        # By Default the x inputs box and y targets box are the same length
        # Any check to the x inputs box reflects the status of the y targets box
        assert self.x_inputs_box.value.length != 0, "Inputs & Targets Data has not been set — call 'add_inputs_and_targets' method first to set training data"
        self.weight = UInt64(0)
        self.bias = UInt64(0)
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Training Loop iteration method will fail if this method is not called first
        self.training_complete = False

        # Do not allow predictions until training is complete
        self.ready_for_training = True

    @abimethod
    def add_inputs_and_targets(
        self, 
        parts_x_inputs: DynamicArray[arc4.UInt64], 
        parts_y_targets: DynamicArray[arc4.UInt64],
        mbr_payment: gtxn.PaymentTransaction
    ) -> None:
        ''' Add Inputs & Targets to Box Storage, Take in an MBR Payment and Refund Excess if there is any'''
        
        # Get the current box content with inputs & targets
        current_x_inputs = self.x_inputs_box.get(default=Data()).copy()
        current_y_targets = self.y_targets_box.get(default=Data()).copy()
        
        # Inputs & Targets need to match in length
        # Each X Input requires a respective Y Target
        assert parts_x_inputs.length == parts_y_targets.length
        parts_length = parts_x_inputs.length

        # The amount of new input & target pairs to resize the box
        size_to_add_to_box = parts_x_inputs.bytes.length

        # Get the current minimum balance requirement for the contract
        # before we start creating/resizing boxes, we send the difference
        # after this is done back to the developer
        pre_mbr = Global.current_application_address.min_balance

        # If the current length of input/target pairs is 0, box has not been created yet
        # Create with size of inputs/targets parts in args
        if current_x_inputs.length == 0:
            # We assign the result to x_1 and x_2 so the compiler doesn't complain
            x_1 = self.x_inputs_box.create(size=size_to_add_to_box)
            x_2 = self.y_targets_box.create(size=size_to_add_to_box)

        else:
            # Get the current length in bytes of box values
            # The lengths of the x_inputs and y_targets boxes will always be the same
            # because of assertion on line 43
            # Since we use ARC4 dynamic arrays, we must account for variable length encoding, we use byte length instead of array length (number of bytes vs items)
            current_byte_size_of_box_bytes = current_x_inputs.bytes.length
            new_size = size_to_add_to_box + current_byte_size_of_box_bytes

            # Resize the box to allow extending the dynamic array with new inputs & targets
            self.x_inputs_box.resize(new_size=new_size)
            self.y_targets_box.resize(new_size=new_size)

        # The boxes for x inputs and y_targets have been created/resized at this point,
        # and new minimum balance requirement is set, we can define the new minimum balance requirement
        post_mbr = Global.current_application_address.min_balance

        # Extend* the current x_inputs and y_targets (can be empty) with the our new x_inputs and y_targets
        current_x_inputs.extend(parts_x_inputs)
        current_y_targets.extend(parts_y_targets)

        # Reset the box value to this new, extended array, containing previous x_inputs and y_targets
        # and are new x_inputs and y_targets parts
        self.x_inputs_box.value = current_x_inputs.copy()
        self.y_targets_box.value = current_x_inputs.copy()

        # Increment the length of our global tracker of data inputs & targets length
        # AKA How many pairs of inputs & targets are we using
        # This number is needed to average deltas in each training loop
        self.length_data += parts_length

        # Lets send the user back any excess Algorand they provided to ensure we maintained the minimum balance requirement
        # Deduct the different of the new MBR requirement and the previous MBR requirement from the total amount sent from the
        # original mbr payment. The transaction fee is covered in the outer transaction.
        mbr_needed = post_mbr - pre_mbr
        itxn.Payment(
            receiver=Txn.sender,
            amount=mbr_payment.amount - mbr_needed,
        ).submit()

    @abimethod
    def run_a_training_loop(self) -> None:
        '''Run a singular training loop with our saved data set and update our global weight and bias'''
        # Dev must call prepare data within contract boxes first via 'add_inputs_and_targets_method' 
        # Then set the contract state to ready for training by calling 'prime_training'
        assert self.ready_for_training, \
        "If you have already added your inputs & targets via 'add_inputs_and_targets_method' " \
        "you must first call 'set_ready_for_training' method before calling this 'run_a_training_loop' method" 

        # Get the epochs completed so far, we do not exceed the training loops (epochs) set in the 'prime_training' method
        current_epoch = self.epochs_completed
        assert current_epoch < self.epochs, "All epochs for training have already been completed, no more training loops needed"

        # Get all data for x_inputs and y_targets
        x_inputs = self.x_inputs_box.value.copy()
        y_targets = self.y_targets_box.value.copy()


        # Initialize our deltas for weight and bias for this training loop
        # These keep track of error sums to apply to our global weight and bias
        delta_weight = UInt64(0)
        delta_bias = UInt64(0)

        # There is no zip method currently available in puyapy
        # Although they did add 'in' operator for iterating through dynamic arrays
        # we will need to use urange and indexing for now

        for i in urange(x_inputs.length):
            # Get each x, y pair in our training data for this epoch
            x = x_inputs[i]
            y = y_targets[i]

            # Guess the value of Y using equation of a line with our weight and bias; y = mx + b, where m = weight and b = bias
            m = self.weight
            b = self.bias
            guessed_y = m * x.as_uint64() + b

            # Check how wrong we were and get the difference of the guessed y target and the actual y target
            y_error = guessed_y - y.as_uint64()

            delta_weight += y_error * m
            delta_bias += y_error

        # Average our delta weight and delta bias by the length of data
        delta_weight //= self.length_data
        delta_bias //= self.length_data

        # Update our global weight and bias by deducting the difference of the product of learning rate and respective delta
        self.weight *= self.learning_rate * delta_weight
        self.bias *= self.learning_rate * delta_bias

        # Increment epoch by 1, if we have completed all epochs set training to complete
        # We can now call the predict method
        self.epochs_completed += 1
        if self.epochs_completed == self.epochs:
            self.training_complete = True

    @abimethod
    def predict(self, x: UInt64) -> UInt64:
        '''Use our trained weight and bias into a f(x) for equation of a line to predict a target y for some value of x'''
        # Optional: Check if we have completed training the model
        assert self.training_complete == True, "Model has not been fully trained"

        # y = mx + b
        return self.weight * x + self.bias
        
