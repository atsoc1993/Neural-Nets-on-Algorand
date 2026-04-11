from algopy import ARC4Contract, UInt64, BigUInt, arc4, Box, gtxn, Global, Txn, itxn, urange, subroutine, ensure_budget, OpUpFeeSource, op
from algopy.arc4 import abimethod, DynamicArray

Data = DynamicArray[arc4.UInt64]


class LogisticRegressionModel(ARC4Contract):
    def __init__(self) -> None:
        self.age_weight_magnitude = UInt64(0)
        self.age_weight_is_negative = False

        self.gender_weight_magnitude = UInt64(0)
        self.gender_weight_is_negative = False

        self.bias_magnitude = UInt64(0)
        self.bias_is_negative = False

        self.epochs = UInt64(0)
        self.epochs_completed = UInt64(0)

        self.learning_rate = UInt64(0)
        self.scale_factor = UInt64(0)

        self.age_inputs_box = Box(Data, key="ages")
        self.gender_inputs_box = Box(Data, key="genders")
        self.y_targets_box = Box(Data, key="targets")

        self.length_data = UInt64(0)
        self.ready_for_training = False
        self.training_complete = False

        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False

        self.fees_used = UInt64(0)

    @abimethod
    def add_inputs_and_targets(
        self,
        parts_age_inputs: Data,
        parts_gender_inputs: Data,
        parts_y_targets: Data,
        mbr_payment: gtxn.PaymentTransaction
    ) -> None:
        assert self.ready_for_training == False
        assert parts_age_inputs.length == parts_gender_inputs.length
        assert parts_age_inputs.length == parts_y_targets.length

        current_age_inputs = self.age_inputs_box.get(default=Data()).copy()
        current_gender_inputs = self.gender_inputs_box.get(default=Data()).copy()
        current_y_targets = self.y_targets_box.get(default=Data()).copy()

        parts_length = parts_age_inputs.length

        pre_mbr = Global.current_application_address.min_balance

        if current_age_inputs.length == 0:
            self.age_inputs_box.create(size=parts_age_inputs.bytes.length)
            self.gender_inputs_box.create(size=parts_gender_inputs.bytes.length)
            self.y_targets_box.create(size=parts_y_targets.bytes.length)
        else:
            self.age_inputs_box.resize(new_size=current_age_inputs.bytes.length + parts_age_inputs.bytes.length)
            self.gender_inputs_box.resize(new_size=current_gender_inputs.bytes.length + parts_gender_inputs.bytes.length)
            self.y_targets_box.resize(new_size=current_y_targets.bytes.length + parts_y_targets.bytes.length)

        post_mbr = Global.current_application_address.min_balance

        current_age_inputs.extend(parts_age_inputs)
        current_gender_inputs.extend(parts_gender_inputs)
        current_y_targets.extend(parts_y_targets)

        self.age_inputs_box.value = current_age_inputs.copy()
        self.gender_inputs_box.value = current_gender_inputs.copy()
        self.y_targets_box.value = current_y_targets.copy()

        self.length_data += parts_length

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
        assert self.age_inputs_box.value.length != 0, "Inputs & Targets Data has not been set"
        assert self.gender_inputs_box.value.length != 0, "Inputs & Targets Data has not been set"
        assert self.y_targets_box.value.length != 0, "Inputs & Targets Data has not been set"
        assert self.length_data > 0
        assert epochs > 0
        assert learning_rate > 0
        assert scale_factor > 0

        self.age_weight_magnitude = self.scale_factor_seed(scale_factor, UInt64(1))
        self.age_weight_is_negative = True

        self.gender_weight_magnitude = self.scale_factor_seed(scale_factor, UInt64(2))
        self.gender_weight_is_negative = False

        self.bias_magnitude = self.scale_factor_seed(scale_factor, UInt64(3))
        self.bias_is_negative = True

        self.scale_factor = scale_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epochs_completed = UInt64(0)

        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False

        self.training_complete = False
        self.ready_for_training = True

        self.fees_used = UInt64(0)

    @abimethod
    def run_training_loops(self) -> None:
        assert self.ready_for_training, \
        "If you have already added your inputs & targets via 'add_inputs_and_targets' " \
        "you must first call 'prime_training' before calling this method"

        ensure_budget(700 * 200, OpUpFeeSource.GroupCredit)

        if not self.calculated_budget_needed:
            self.discover_and_store_budget()

        epochs_remaining = self.epochs - self.epochs_completed
        max_epochs_runnable = Global.opcode_budget() // self.extra_budget_needed

        if max_epochs_runnable > epochs_remaining:
            epochs_to_run = epochs_remaining
        else:
            epochs_to_run = max_epochs_runnable

        for epoch in urange(epochs_to_run):
            self.run_a_training_loop()

    @subroutine
    def discover_and_store_budget(self) -> None:
        pre_budget_remaining = Global.opcode_budget()

        self.run_a_training_loop()

        post_budget_remaining = Global.opcode_budget()

        budget_cost_per_iteration = pre_budget_remaining - post_budget_remaining
        self.extra_budget_needed = budget_cost_per_iteration
        self.calculated_budget_needed = True

    @subroutine
    def run_a_training_loop(self) -> None:
        assert self.epochs_completed < self.epochs, "All epochs for training have already been completed"

        delta_age_weight_magnitude = UInt64(0)
        delta_age_weight_is_negative = False

        delta_gender_weight_magnitude = UInt64(0)
        delta_gender_weight_is_negative = False

        delta_bias_magnitude = UInt64(0)
        delta_bias_is_negative = False

        age_inputs = self.age_inputs_box.value.copy()
        gender_inputs = self.gender_inputs_box.value.copy()
        y_targets = self.y_targets_box.value.copy()

        for i in urange(self.length_data):
            age = age_inputs[i].as_uint64()
            gender = gender_inputs[i].as_uint64()
            y = y_targets[i].as_uint64()

            guessed_y_magnitude, guessed_y_is_negative = self.predict_signed_logit(
                self.age_weight_magnitude,
                self.age_weight_is_negative,
                self.gender_weight_magnitude,
                self.gender_weight_is_negative,
                self.bias_magnitude,
                self.bias_is_negative,
                age,
                gender
            )

            activated_y_prediction = self.hard_sigmoid(
                guessed_y_magnitude,
                guessed_y_is_negative
            )

            y_error_magnitude, y_error_is_negative = self.signed_sub(
                activated_y_prediction,
                False,
                y,
                False
            )

            grad_age_weight_magnitude = self.scale_down_product(y_error_magnitude, age)
            grad_age_weight_is_negative = y_error_is_negative

            delta_age_weight_magnitude, delta_age_weight_is_negative = self.signed_add(
                delta_age_weight_magnitude,
                delta_age_weight_is_negative,
                grad_age_weight_magnitude,
                grad_age_weight_is_negative
            )

            grad_gender_weight_magnitude = self.scale_down_product(y_error_magnitude, gender)
            grad_gender_weight_is_negative = y_error_is_negative

            delta_gender_weight_magnitude, delta_gender_weight_is_negative = self.signed_add(
                delta_gender_weight_magnitude,
                delta_gender_weight_is_negative,
                grad_gender_weight_magnitude,
                grad_gender_weight_is_negative
            )

            delta_bias_magnitude, delta_bias_is_negative = self.signed_add(
                delta_bias_magnitude,
                delta_bias_is_negative,
                y_error_magnitude,
                y_error_is_negative
            )

        delta_age_weight_magnitude = delta_age_weight_magnitude // self.length_data
        delta_gender_weight_magnitude = delta_gender_weight_magnitude // self.length_data
        delta_bias_magnitude = delta_bias_magnitude // self.length_data

        age_weight_update_amount = self.scale_down_product(self.learning_rate, delta_age_weight_magnitude)
        gender_weight_update_amount = self.scale_down_product(self.learning_rate, delta_gender_weight_magnitude)
        bias_update_amount = self.scale_down_product(self.learning_rate, delta_bias_magnitude)

        self.age_weight_magnitude, self.age_weight_is_negative = self.signed_sub(
            self.age_weight_magnitude,
            self.age_weight_is_negative,
            age_weight_update_amount,
            delta_age_weight_is_negative
        )

        self.gender_weight_magnitude, self.gender_weight_is_negative = self.signed_sub(
            self.gender_weight_magnitude,
            self.gender_weight_is_negative,
            gender_weight_update_amount,
            delta_gender_weight_is_negative
        )

        self.bias_magnitude, self.bias_is_negative = self.signed_sub(
            self.bias_magnitude,
            self.bias_is_negative,
            bias_update_amount,
            delta_bias_is_negative
        )

        self.epochs_completed += 1

        if self.epochs_completed == self.epochs:
            self.training_complete = True
            self.ready_for_training = False

        self.fees_used += Txn.fee

    @subroutine
    def predict_signed_logit(
        self,
        age_weight_magnitude: UInt64,
        age_weight_is_negative: bool,
        gender_weight_magnitude: UInt64,
        gender_weight_is_negative: bool,
        bias_magnitude: UInt64,
        bias_is_negative: bool,
        age: UInt64,
        gender: UInt64
    ) -> tuple[UInt64, bool]:
        weighted_age_magnitude = self.scale_down_product(age_weight_magnitude, age)
        weighted_age_is_negative = age_weight_is_negative

        weighted_gender_magnitude = self.scale_down_product(gender_weight_magnitude, gender)
        weighted_gender_is_negative = gender_weight_is_negative

        combined_magnitude, combined_is_negative = self.signed_add(
            weighted_age_magnitude,
            weighted_age_is_negative,
            weighted_gender_magnitude,
            weighted_gender_is_negative
        )

        return self.signed_add(
            combined_magnitude,
            combined_is_negative,
            bias_magnitude,
            bias_is_negative
        )

    @abimethod
    def predict_logit(self, age: UInt64, gender: UInt64) -> tuple[UInt64, bool]:
        assert self.training_complete == True, "Model has not been fully trained"

        return self.predict_signed_logit(
            self.age_weight_magnitude,
            self.age_weight_is_negative,
            self.gender_weight_magnitude,
            self.gender_weight_is_negative,
            self.bias_magnitude,
            self.bias_is_negative,
            age,
            gender
        )

    @abimethod
    def predict(self, age: UInt64, gender: UInt64) -> UInt64:
        assert self.training_complete == True, "Model has not been fully trained"

        predicted_logit_magnitude, predicted_logit_is_negative = self.predict_signed_logit(
            self.age_weight_magnitude,
            self.age_weight_is_negative,
            self.gender_weight_magnitude,
            self.gender_weight_is_negative,
            self.bias_magnitude,
            self.bias_is_negative,
            age,
            gender
        )

        return self.hard_sigmoid(
            predicted_logit_magnitude,
            predicted_logit_is_negative
        )

    @abimethod
    def clear_data(self) -> None:
        self.age_inputs_box.value = Data()
        self.gender_inputs_box.value = Data()
        self.y_targets_box.value = Data()

        self.length_data = UInt64(0)

        self.age_weight_magnitude = UInt64(0)
        self.age_weight_is_negative = False

        self.gender_weight_magnitude = UInt64(0)
        self.gender_weight_is_negative = False

        self.bias_magnitude = UInt64(0)
        self.bias_is_negative = False

        self.epochs = UInt64(0)
        self.epochs_completed = UInt64(0)
        self.learning_rate = UInt64(0)
        self.scale_factor = UInt64(0)

        self.ready_for_training = False
        self.training_complete = False

        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False
        self.fees_used = UInt64(0)

    @subroutine
    def scale_factor_seed(self, scale_factor: UInt64, divisor: UInt64) -> UInt64:
        seeded = scale_factor // (UInt64(10) * divisor)
        if seeded == 0:
            return UInt64(1)
        return seeded

    @subroutine
    def scale_down_product(self, a: UInt64, b: UInt64) -> UInt64:
        return op.btoi(((BigUInt(a) * b) // self.scale_factor).bytes)

    @subroutine
    def hard_sigmoid(self, magnitude: UInt64, is_negative: bool) -> UInt64:
        four_scale = self.scale_factor * UInt64(4)
        half_scale = self.scale_factor // UInt64(2)

        if is_negative:
            if magnitude >= four_scale:
                return UInt64(0)
            return half_scale - (magnitude // UInt64(8))

        if magnitude >= four_scale:
            return self.scale_factor

        value = half_scale + (magnitude // UInt64(8))
        if value > self.scale_factor:
            return self.scale_factor
        return value

    @subroutine
    def signed_add(
        self,
        a_magnitude: UInt64,
        a_is_negative: bool,
        b_magnitude: UInt64,
        b_is_negative: bool
    ) -> tuple[UInt64, bool]:
        if a_is_negative == b_is_negative:
            return a_magnitude + b_magnitude, a_is_negative

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
        return self.signed_add(
            a_magnitude,
            a_is_negative,
            b_magnitude,
            not b_is_negative
        )