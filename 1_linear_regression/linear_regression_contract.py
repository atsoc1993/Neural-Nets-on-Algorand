from algopy import ARC4Contract, UInt64, BigUInt, arc4, Box, gtxn, Global, Txn, itxn, urange, subroutine, ensure_budget, OpUpFeeSource, op
from algopy.arc4 import abimethod, DynamicArray

Data = DynamicArray[arc4.UInt64]

class LinearRegressionModel(ARC4Contract):
    def __init__(self) -> None:
        # Weight and Bias are stored as magnitude + sign flag
        # Both are fixed-point scaled values
        self.weight_magnitude = UInt64(0)
        self.weight_is_negative = False
        self.bias_magnitude = UInt64(0)
        self.bias_is_negative = False

        self.epochs = UInt64(0)
        self.epochs_completed = UInt64(0)

        # learning_rate is also fixed-point scaled
        self.learning_rate = UInt64(0)

        # Must match your off-chain scale
        self.scale_factor = UInt64(0)

        self.x_inputs_box = Box(DynamicArray[arc4.UInt64], key='inputs')
        self.y_targets_box = Box(DynamicArray[arc4.UInt64], key='targets')
        self.length_data = UInt64(0)
        self.ready_for_training = False
        self.training_complete = False

        # Cached budget discovery
        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False

        # Optional bookkeeping
        self.fees_used = UInt64(0)
        
    @abimethod
    def add_inputs_and_targets(
        self, 
        parts_x_inputs: DynamicArray[arc4.UInt64], 
        parts_y_targets: DynamicArray[arc4.UInt64],
        mbr_payment: gtxn.PaymentTransaction
    ) -> None:
        ''' Add Inputs & Targets to Box Storage, Take in an MBR Payment and Refund Excess if there is any'''
        
        assert self.ready_for_training == False

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
            x_1 = self.x_inputs_box.create(size=size_to_add_to_box)
            x_2 = self.y_targets_box.create(size=size_to_add_to_box)
        else:
            # Get the current length in bytes of box values
            # The lengths of the x_inputs and y_targets boxes will always be the same
            # because of assertion above
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
        # and our new x_inputs and y_targets parts
        self.x_inputs_box.value = current_x_inputs.copy()
        self.y_targets_box.value = current_y_targets.copy()

        # Increment the length of our global tracker of data inputs & targets length
        # AKA How many pairs of inputs & targets are we using
        # This number is needed to average deltas in each training loop
        self.length_data += parts_length

        # Lets send the user back any excess Algorand they provided to ensure we maintained the minimum balance requirement
        # Deduct the difference of the new MBR requirement and the previous MBR requirement from the total amount sent from the
        # original mbr payment. The transaction fee is covered in the outer transaction.
        mbr_needed = post_mbr - pre_mbr
        refund_amount = mbr_payment.amount - mbr_needed

        if refund_amount > 0:
            itxn.Payment(
                receiver=Txn.sender,
                amount=refund_amount,
            ).submit()

        self.fees_used += Txn.fee

    @abimethod
    def prime_training(
        self,
        scale_factor: UInt64,
        learning_rate: UInt64,
        epochs: UInt64
    ) -> None:
        '''Reset Weight & Bias and set the intended epochs and learning rate for new training phase'''

        # By Default the x inputs box and y targets box are the same length
        # Any check to the x inputs box reflects the status of the y targets box
        assert self.x_inputs_box.value.length != 0, "Inputs & Targets Data has not been set — call 'add_inputs_and_targets' method first to set training data"
        assert self.length_data > 0
        assert epochs > 0
        assert learning_rate > 0

        self.weight_magnitude = UInt64(0)
        self.weight_is_negative = False
        self.bias_magnitude = UInt64(0)
        self.bias_is_negative = False

        assert scale_factor < UInt64((2**64) - 1), "SCALE FACTOR TOO LARGE"
        self.scale_factor = scale_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epochs_completed = UInt64(0)

        # Reset cached budget
        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False

        # Training Loop iteration method will fail if this method is not called first
        self.training_complete = False

        # Do not allow predictions until training is complete
        self.ready_for_training = True

        self.fees_used = UInt64(0)

    @abimethod
    def run_training_loops(self) -> None:
        assert self.ready_for_training, \
        "If you have already added your inputs & targets via 'add_inputs_and_targets_method' " \
        "you must first call 'prime_training' method before calling this 'run_training_loops' method" 
        
        # By Default use max inner transactions
        ensure_budget(700 * 200, OpUpFeeSource.GroupCredit) #

        # Only discover needed budget once
        if not self.calculated_budget_needed:
            self.discover_and_store_budget()

        epochs_remaining = self.epochs - self.epochs_completed
        max_epochs_runnable = Global.opcode_budget() // self.extra_budget_needed
        if max_epochs_runnable > epochs_remaining:
            epochs_to_run = epochs_remaining
        else:
            epochs_to_run = max_epochs_runnable

        for i in urange(epochs_to_run):
            self.run_a_training_loop()

        
    @subroutine
    def discover_and_store_budget(self) -> None:
        # Only used to discover the opcode cost of one training iteration
        pre_budget_remaining = Global.opcode_budget()

        self.run_a_training_loop()

        post_budget_remaining = Global.opcode_budget()

        budget_cost_per_iteration = pre_budget_remaining - post_budget_remaining
        self.extra_budget_needed = budget_cost_per_iteration
        self.calculated_budget_needed = True

    @subroutine
    def run_a_training_loop(self) -> None:
        '''Run a singular training loop with our saved data set and update our global weight and bias'''
        # Dev must call prepare data within contract boxes first via 'add_inputs_and_targets_method' 
        # Then set the contract state to ready for training by calling 'prime_training'

        # We do not exceed the training loops (epochs) set in the 'prime_training' method
        assert self.epochs_completed < self.epochs, "All epochs for training have already been completed, no more training loops needed"

        delta_weight_magnitude, delta_weight_is_negative, delta_bias_magnitude, delta_bias_is_negative = self.run_training_loop_by_start_and_end_index(
            UInt64(0),
            self.length_data
        )

        # Average our delta weight and delta bias by the length of data
        delta_weight_magnitude = delta_weight_magnitude // self.length_data
        delta_bias_magnitude = delta_bias_magnitude // self.length_data

        # Keep the same learning rule
        weight_update_amount = op.btoi(((BigUInt(self.learning_rate) * delta_weight_magnitude) // self.scale_factor).bytes)
        bias_update_amount = op.btoi(((BigUInt(self.learning_rate) * delta_bias_magnitude) // self.scale_factor).bytes)

        self.weight_magnitude, self.weight_is_negative = self.signed_sub(
            self.weight_magnitude,
            self.weight_is_negative,
            weight_update_amount,
            delta_weight_is_negative
        )

        self.bias_magnitude, self.bias_is_negative = self.signed_sub(
            self.bias_magnitude,
            self.bias_is_negative,
            bias_update_amount,
            delta_bias_is_negative
        )

        # Increment epoch by 1, if we have completed all epochs set training to complete
        # We can now call the predict method
        self.epochs_completed += 1

        if self.epochs_completed == self.epochs:
            self.training_complete = True
            self.ready_for_training = False

        self.fees_used += Txn.fee


    @subroutine
    def run_training_loop_by_start_and_end_index(
        self,
        start: UInt64,
        end: UInt64,
    ) -> tuple[UInt64, bool, UInt64, bool]:

        # Initialize our deltas for weight and bias for this training loop
        # These keep track of error sums to apply to our global weight and bias
        delta_weight_magnitude = UInt64(0)
        delta_weight_is_negative = False
        delta_bias_magnitude = UInt64(0)
        delta_bias_is_negative = False

        # Get all data for x_inputs and y_targets
        x_inputs = self.x_inputs_box.value.copy()
        y_targets = self.y_targets_box.value.copy()

        for i in urange(start, end):
            # Get each x, y pair in our training data for this epoch
            x = x_inputs[i].as_uint64()
            y = y_targets[i].as_uint64()

            # Guess the value of Y using equation of a line with our weight and bias; y = mx + b
            weighted_x_magnitude = op.btoi(((BigUInt(self.weight_magnitude) * x) // self.scale_factor).bytes)
            weighted_x_is_negative = self.weight_is_negative

            guessed_y_magnitude, guessed_y_is_negative = self.signed_add(
                weighted_x_magnitude,
                weighted_x_is_negative,
                self.bias_magnitude,
                self.bias_is_negative
            )

            # Check how wrong we were and get the signed difference of the guessed y target and the actual y target
            y_error_magnitude, y_error_is_negative = self.signed_sub(
                guessed_y_magnitude,
                guessed_y_is_negative,
                y,
                False
            )

            # delta_weight += y_error * x
            # Keep the same logic as the float version, but fixed-point scale the product back down once
            grad_weight_magnitude = op.btoi(((BigUInt(y_error_magnitude) * x) // self.scale_factor).bytes)

            grad_weight_is_negative = y_error_is_negative

            delta_weight_magnitude, delta_weight_is_negative = self.signed_add(
                delta_weight_magnitude,
                delta_weight_is_negative,
                grad_weight_magnitude,
                grad_weight_is_negative
            )

            # delta_bias += y_error
            delta_bias_magnitude, delta_bias_is_negative = self.signed_add(
                delta_bias_magnitude,
                delta_bias_is_negative,
                y_error_magnitude,
                y_error_is_negative
            )

        return (
            delta_weight_magnitude,
            delta_weight_is_negative,
            delta_bias_magnitude,
            delta_bias_is_negative
        )

    @subroutine
    def predict_signed_value(
        self,
        weight_magnitude: UInt64,
        weight_is_negative: bool,
        bias_magnitude: UInt64,
        bias_is_negative: bool,
        x: UInt64
    ) -> tuple[UInt64, bool]:
        # y = weight * x + bias
        # weight is scaled and x is scaled, so divide once by scale_factor
        weighted_x_magnitude = op.btoi(((BigUInt(weight_magnitude) * x) // self.scale_factor).bytes)
        weighted_x_is_negative = weight_is_negative

        return self.signed_add(
            weighted_x_magnitude,
            weighted_x_is_negative,
            bias_magnitude,
            bias_is_negative
        )

    @abimethod
    def predict(self, x: UInt64) -> tuple[UInt64, bool]:
        '''Use our trained weight and bias into a f(x) for equation of a line to predict a target y for some value of x'''
        assert self.training_complete == True, "Model has not been fully trained"

        return self.predict_signed_value(
            self.weight_magnitude,
            self.weight_is_negative,
            self.bias_magnitude,
            self.bias_is_negative,
            x
        )
    
    @abimethod
    def clear_data(self) -> None:
        self.x_inputs_box.value = Data()
        self.y_targets_box.value = Data()
        self.length_data = UInt64(0)

        self.weight_magnitude = UInt64(0)
        self.weight_is_negative = False
        self.bias_magnitude = UInt64(0)
        self.bias_is_negative = False

        self.epochs = UInt64(0)
        self.epochs_completed = UInt64(0)
        self.learning_rate = UInt64(0)
        self.ready_for_training = False
        self.training_complete = False

        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False
        self.fees_used = UInt64(0)

    @subroutine
    def signed_add(
        self,
        a_magnitude: UInt64,
        a_is_negative: bool,
        b_magnitude: UInt64,
        b_is_negative: bool
    ) -> tuple[UInt64, bool]:
        # If both values have the same sign, just add magnitudes and keep the sign
        if a_is_negative == b_is_negative:
            return a_magnitude + b_magnitude, a_is_negative

        # If the signs are different, subtract the smaller magnitude from the larger one
        # and keep the sign of the larger magnitude
        if a_magnitude >= b_magnitude:
            return a_magnitude - b_magnitude, a_is_negative

        return b_magnitude - a_magnitude, b_is_negative

    @subroutine
    def signed_sub(
        self,
        a_magnitude: UInt64,
        a_is_negative: bool,
        b_magnitude: UInt64,
        b_is_negative: bool
    ) -> tuple[UInt64, bool]:
        # a - b is the same thing as a + negative(b)
        return self.signed_add(
            a_magnitude,
            a_is_negative,
            b_magnitude,
            not b_is_negative
        )