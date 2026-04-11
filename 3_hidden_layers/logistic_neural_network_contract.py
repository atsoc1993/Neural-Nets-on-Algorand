from algopy import ARC4Contract, UInt64, BigUInt, arc4, Box, gtxn, Global, Txn, itxn, urange, subroutine, ensure_budget, OpUpFeeSource, op
from algopy.arc4 import abimethod, DynamicArray

Data = DynamicArray[arc4.UInt64]


class LogisticNeuralNetwork(ARC4Contract):
    def __init__(self) -> None:
        self.input_feature_count = UInt64(2)

        self.hidden_layers = UInt64(0)
        self.hidden_neurons = UInt64(0)

        self.epochs = UInt64(0)
        self.epochs_completed = UInt64(0)
        self.learning_rate = UInt64(0)
        self.scale_factor = UInt64(0)

        self.length_data = UInt64(0)
        self.ready_for_training = False
        self.training_complete = False

        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False
        self.fees_used = UInt64(0)

        self.age_inputs_box = Box(Data, key="ages")
        self.gender_inputs_box = Box(Data, key="genders")
        self.y_targets_box = Box(Data, key="targets")

        self.first_hidden_weight_magnitudes_box = Box(Data, key="f_h_w_mag")
        self.first_hidden_weight_signs_box = Box(Data, key="f_h_w_sign")

        self.hidden_weight_magnitudes_box = Box(Data, key="h_w_mag")
        self.hidden_weight_signs_box = Box(Data, key="h_w_sign")

        self.hidden_bias_magnitudes_box = Box(Data, key="h_b_mag")
        self.hidden_bias_signs_box = Box(Data, key="h_b_sign")

        self.output_weight_magnitudes_box = Box(Data, key="o_w_mag")
        self.output_weight_signs_box = Box(Data, key="o_w_sign")

        self.output_bias_magnitude = UInt64(0)
        self.output_bias_is_negative = False

    @abimethod
    def add_inputs_and_targets(
        self,
        parts_age_inputs: Data,
        parts_gender_inputs: Data,
        parts_y_targets: Data,
        mbr_payment: gtxn.PaymentTransaction,
    ) -> None:
        assert self.ready_for_training == False
        assert parts_age_inputs.length == parts_gender_inputs.length
        assert parts_age_inputs.length == parts_y_targets.length

        current_ages = self.age_inputs_box.get(default=Data()).copy()
        current_genders = self.gender_inputs_box.get(default=Data()).copy()
        current_targets = self.y_targets_box.get(default=Data()).copy()

        parts_length = parts_age_inputs.length

        pre_mbr = Global.current_application_address.min_balance

        if current_ages.length == 0:
            self.age_inputs_box.create(size=parts_age_inputs.bytes.length)
            self.gender_inputs_box.create(size=parts_gender_inputs.bytes.length)
            self.y_targets_box.create(size=parts_y_targets.bytes.length)
        else:
            self.age_inputs_box.resize(new_size=current_ages.bytes.length + parts_age_inputs.bytes.length)
            self.gender_inputs_box.resize(new_size=current_genders.bytes.length + parts_gender_inputs.bytes.length)
            self.y_targets_box.resize(new_size=current_targets.bytes.length + parts_y_targets.bytes.length)

        post_mbr = Global.current_application_address.min_balance

        current_ages.extend(parts_age_inputs)
        current_genders.extend(parts_gender_inputs)
        current_targets.extend(parts_y_targets)

        self.age_inputs_box.value = current_ages.copy()
        self.gender_inputs_box.value = current_genders.copy()
        self.y_targets_box.value = current_targets.copy()

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
        epochs: UInt64,
        hidden_layers: UInt64,
        hidden_neurons: UInt64,
        mbr_payment: gtxn.PaymentTransaction
    ) -> None:
        assert self.age_inputs_box.value.length != 0, "Inputs & Targets Data has not been set"
        assert self.length_data > 0
        assert epochs > 0
        assert learning_rate > 0
        assert hidden_layers > 0
        assert hidden_neurons > 0

        self.scale_factor = scale_factor
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.epochs_completed = UInt64(0)

        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons

        self.output_bias_magnitude = UInt64(0)
        self.output_bias_is_negative = False

        self.extra_budget_needed = UInt64(0)
        self.calculated_budget_needed = False

        self.training_complete = False
        self.ready_for_training = True
        self.fees_used = UInt64(0)

        ensure_budget(hidden_layers * hidden_neurons * 700, fee_source=OpUpFeeSource.GroupCredit)

        pre_mbr = Global.current_application_address.min_balance

        self.initialize_parameters()

        post_mbr = Global.current_application_address.min_balance

        mbr_needed = post_mbr - pre_mbr
        refund_amount = mbr_payment.amount - mbr_needed

        if refund_amount > 0:
            itxn.Payment(
                receiver=Txn.sender,
                amount=refund_amount,
            ).submit()

    @abimethod
    def run_training_loops(self) -> None:
        assert self.ready_for_training, \
        "If you have already added your inputs & targets via 'add_inputs_and_targets' " \
        "you must first call 'prime_training' before calling this method"

        ensure_budget(700 * 248, OpUpFeeSource.GroupCredit)

        if not self.calculated_budget_needed:
            self.discover_and_store_budget()

        epochs_remaining = self.epochs - self.epochs_completed
        max_epochs_runnable = Global.opcode_budget() // self.extra_budget_needed

        if max_epochs_runnable > epochs_remaining:
            epochs_to_run = epochs_remaining
        else:
            epochs_to_run = max_epochs_runnable

        for epoch in urange(epochs_to_run - 10000):
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

        hidden_state_count = self.hidden_layers * self.hidden_neurons
        first_hidden_weight_count = self.hidden_neurons * self.input_feature_count

        hidden_weight_count = UInt64(0)
        if self.hidden_layers > 1:
            hidden_weight_count = (self.hidden_layers - UInt64(1)) * self.hidden_neurons * self.hidden_neurons

        age_inputs = self.age_inputs_box.value.copy()
        gender_inputs = self.gender_inputs_box.value.copy()
        y_targets = self.y_targets_box.value.copy()

        first_hidden_weight_magnitudes = self.first_hidden_weight_magnitudes_box.value.copy()
        first_hidden_weight_signs = self.first_hidden_weight_signs_box.value.copy()

        hidden_weight_magnitudes = self.hidden_weight_magnitudes_box.value.copy()
        hidden_weight_signs = self.hidden_weight_signs_box.value.copy()

        hidden_bias_magnitudes = self.hidden_bias_magnitudes_box.value.copy()
        hidden_bias_signs = self.hidden_bias_signs_box.value.copy()

        output_weight_magnitudes = self.output_weight_magnitudes_box.value.copy()
        output_weight_signs = self.output_weight_signs_box.value.copy()

        delta_first_hidden_weight_magnitudes = self.make_zero_array(first_hidden_weight_count)
        delta_first_hidden_weight_signs = self.make_zero_array(first_hidden_weight_count)

        delta_hidden_weight_magnitudes_arr = self.make_zero_array(hidden_weight_count)
        delta_hidden_weight_signs_arr = self.make_zero_array(hidden_weight_count)

        delta_hidden_bias_magnitudes = self.make_zero_array(hidden_state_count)
        delta_hidden_bias_signs = self.make_zero_array(hidden_state_count)

        delta_output_weight_magnitudes = self.make_zero_array(self.hidden_neurons)
        delta_output_weight_signs = self.make_zero_array(self.hidden_neurons)

        delta_output_bias_magnitude = UInt64(0)
        delta_output_bias_is_negative = False

        hidden_preactivated_magnitudes = self.make_zero_array(hidden_state_count)
        hidden_preactivated_signs = self.make_zero_array(hidden_state_count)
        hidden_activations = self.make_zero_array(hidden_state_count)

        hidden_error_magnitudes = self.make_zero_array(hidden_state_count)
        hidden_error_signs = self.make_zero_array(hidden_state_count)

        for data_index in urange(self.length_data):
            self.zero_existing_array(hidden_preactivated_magnitudes, hidden_state_count)
            self.zero_existing_array(hidden_preactivated_signs, hidden_state_count)
            self.zero_existing_array(hidden_activations, hidden_state_count)
            self.zero_existing_array(hidden_error_magnitudes, hidden_state_count)
            self.zero_existing_array(hidden_error_signs, hidden_state_count)

            age = age_inputs[data_index].as_uint64()
            gender = gender_inputs[data_index].as_uint64()
            y = y_targets[data_index].as_uint64()

            for layer in urange(self.hidden_layers):
                for neuron in urange(self.hidden_neurons):
                    bias_index = self.hidden_bias_index(layer, neuron)
                    preactivation_magnitude = hidden_bias_magnitudes[bias_index].as_uint64()
                    preactivation_is_negative = self.uint_to_bool(hidden_bias_signs[bias_index].as_uint64())

                    if layer == 0:
                        first_weight_index_age = self.first_hidden_weight_index(neuron, UInt64(0))
                        first_weight_index_gender = self.first_hidden_weight_index(neuron, UInt64(1))

                        preactivation_magnitude, preactivation_is_negative = self.accumulate_signed_scaled_product(
                            preactivation_magnitude,
                            preactivation_is_negative,
                            first_hidden_weight_magnitudes[first_weight_index_age].as_uint64(),
                            self.uint_to_bool(first_hidden_weight_signs[first_weight_index_age].as_uint64()),
                            age,
                        )

                        preactivation_magnitude, preactivation_is_negative = self.accumulate_signed_scaled_product(
                            preactivation_magnitude,
                            preactivation_is_negative,
                            first_hidden_weight_magnitudes[first_weight_index_gender].as_uint64(),
                            self.uint_to_bool(first_hidden_weight_signs[first_weight_index_gender].as_uint64()),
                            gender,
                        )
                    else:
                        for prev_neuron in urange(self.hidden_neurons):
                            weight_index = self.hidden_weight_index(layer, neuron, prev_neuron)
                            prev_activation_index = self.hidden_bias_index(layer - UInt64(1), prev_neuron)

                            preactivation_magnitude, preactivation_is_negative = self.accumulate_signed_scaled_product(
                                preactivation_magnitude,
                                preactivation_is_negative,
                                hidden_weight_magnitudes[weight_index].as_uint64(),
                                self.uint_to_bool(hidden_weight_signs[weight_index].as_uint64()),
                                hidden_activations[prev_activation_index].as_uint64(),
                            )

                    hidden_preactivated_magnitudes[bias_index] = arc4.UInt64(preactivation_magnitude)
                    hidden_preactivated_signs[bias_index] = arc4.UInt64(self.bool_to_uint(preactivation_is_negative))
                    hidden_activations[bias_index] = arc4.UInt64(
                        self.relu(preactivation_magnitude, preactivation_is_negative)
                    )

            z_out_magnitude = self.output_bias_magnitude
            z_out_is_negative = self.output_bias_is_negative

            for neuron in urange(self.hidden_neurons):
                last_hidden_index = self.hidden_bias_index(self.hidden_layers - UInt64(1), neuron)
                z_out_magnitude, z_out_is_negative = self.accumulate_signed_scaled_product(
                    z_out_magnitude,
                    z_out_is_negative,
                    output_weight_magnitudes[neuron].as_uint64(),
                    self.uint_to_bool(output_weight_signs[neuron].as_uint64()),
                    hidden_activations[last_hidden_index].as_uint64(),
                )

            y_prediction = self.hard_sigmoid(z_out_magnitude, z_out_is_negative)

            y_error_magnitude, y_error_is_negative = self.signed_sub(
                y_prediction,
                False,
                y,
                False,
            )

            for neuron in urange(self.hidden_neurons):
                last_hidden_index = self.hidden_bias_index(self.hidden_layers - UInt64(1), neuron)

                grad_output_weight_magnitude = self.scale_down_product(
                    y_error_magnitude,
                    hidden_activations[last_hidden_index].as_uint64(),
                )
                grad_output_weight_is_negative = y_error_is_negative

                new_mag, new_sign = self.signed_add(
                    delta_output_weight_magnitudes[neuron].as_uint64(),
                    self.uint_to_bool(delta_output_weight_signs[neuron].as_uint64()),
                    grad_output_weight_magnitude,
                    grad_output_weight_is_negative,
                )
                delta_output_weight_magnitudes[neuron] = arc4.UInt64(new_mag)
                delta_output_weight_signs[neuron] = arc4.UInt64(self.bool_to_uint(new_sign))

            delta_output_bias_magnitude, delta_output_bias_is_negative = self.signed_add(
                delta_output_bias_magnitude,
                delta_output_bias_is_negative,
                y_error_magnitude,
                y_error_is_negative,
            )

            for neuron in urange(self.hidden_neurons):
                last_index = self.hidden_bias_index(self.hidden_layers - UInt64(1), neuron)
                relu_deriv = self.relu_derivative(
                    hidden_preactivated_magnitudes[last_index].as_uint64(),
                    self.uint_to_bool(hidden_preactivated_signs[last_index].as_uint64()),
                )

                base_error_magnitude = self.scale_down_product(
                    y_error_magnitude,
                    output_weight_magnitudes[neuron].as_uint64(),
                )
                base_error_is_negative = self.xor_signs(
                    y_error_is_negative,
                    self.uint_to_bool(output_weight_signs[neuron].as_uint64()),
                )

                if relu_deriv == UInt64(1):
                    hidden_error_magnitudes[last_index] = arc4.UInt64(base_error_magnitude)
                    hidden_error_signs[last_index] = arc4.UInt64(self.bool_to_uint(base_error_is_negative))

            if self.hidden_layers > 1:
                for reverse_offset in urange(self.hidden_layers - UInt64(1)):
                    reverse_layer = (self.hidden_layers - UInt64(2)) - reverse_offset

                    for neuron in urange(self.hidden_neurons):
                        weighted_error_sum_magnitude = UInt64(0)
                        weighted_error_sum_is_negative = False

                        for next_neuron in urange(self.hidden_neurons):
                            next_error_index = self.hidden_bias_index(reverse_layer + UInt64(1), next_neuron)
                            next_weight_index = self.hidden_weight_index(reverse_layer + UInt64(1), next_neuron, neuron)

                            product_magnitude = self.scale_down_product(
                                hidden_error_magnitudes[next_error_index].as_uint64(),
                                hidden_weight_magnitudes[next_weight_index].as_uint64(),
                            )
                            product_is_negative = self.xor_signs(
                                self.uint_to_bool(hidden_error_signs[next_error_index].as_uint64()),
                                self.uint_to_bool(hidden_weight_signs[next_weight_index].as_uint64()),
                            )

                            weighted_error_sum_magnitude, weighted_error_sum_is_negative = self.signed_add(
                                weighted_error_sum_magnitude,
                                weighted_error_sum_is_negative,
                                product_magnitude,
                                product_is_negative,
                            )

                        current_index = self.hidden_bias_index(reverse_layer, neuron)
                        relu_deriv = self.relu_derivative(
                            hidden_preactivated_magnitudes[current_index].as_uint64(),
                            self.uint_to_bool(hidden_preactivated_signs[current_index].as_uint64()),
                        )

                        if relu_deriv == UInt64(1):
                            hidden_error_magnitudes[current_index] = arc4.UInt64(weighted_error_sum_magnitude)
                            hidden_error_signs[current_index] = arc4.UInt64(self.bool_to_uint(weighted_error_sum_is_negative))

            for layer in urange(self.hidden_layers):
                for neuron in urange(self.hidden_neurons):
                    error_index = self.hidden_bias_index(layer, neuron)
                    hidden_error_magnitude = hidden_error_magnitudes[error_index].as_uint64()
                    hidden_error_is_negative = self.uint_to_bool(hidden_error_signs[error_index].as_uint64())

                    if layer == 0:
                        age_weight_delta_index = self.first_hidden_weight_index(neuron, UInt64(0))
                        gender_weight_delta_index = self.first_hidden_weight_index(neuron, UInt64(1))

                        grad_age_magnitude = self.scale_down_product(hidden_error_magnitude, age)
                        grad_gender_magnitude = self.scale_down_product(hidden_error_magnitude, gender)

                        new_mag, new_sign = self.signed_add(
                            delta_first_hidden_weight_magnitudes[age_weight_delta_index].as_uint64(),
                            self.uint_to_bool(delta_first_hidden_weight_signs[age_weight_delta_index].as_uint64()),
                            grad_age_magnitude,
                            hidden_error_is_negative,
                        )
                        delta_first_hidden_weight_magnitudes[age_weight_delta_index] = arc4.UInt64(new_mag)
                        delta_first_hidden_weight_signs[age_weight_delta_index] = arc4.UInt64(self.bool_to_uint(new_sign))

                        new_mag, new_sign = self.signed_add(
                            delta_first_hidden_weight_magnitudes[gender_weight_delta_index].as_uint64(),
                            self.uint_to_bool(delta_first_hidden_weight_signs[gender_weight_delta_index].as_uint64()),
                            grad_gender_magnitude,
                            hidden_error_is_negative,
                        )
                        delta_first_hidden_weight_magnitudes[gender_weight_delta_index] = arc4.UInt64(new_mag)
                        delta_first_hidden_weight_signs[gender_weight_delta_index] = arc4.UInt64(self.bool_to_uint(new_sign))
                    else:
                        for prev_neuron in urange(self.hidden_neurons):
                            weight_delta_index = self.hidden_weight_index(layer, neuron, prev_neuron)
                            prev_activation_index = self.hidden_bias_index(layer - UInt64(1), prev_neuron)

                            grad_hidden_weight_magnitude = self.scale_down_product(
                                hidden_error_magnitude,
                                hidden_activations[prev_activation_index].as_uint64(),
                            )

                            new_mag, new_sign = self.signed_add(
                                delta_hidden_weight_magnitudes_arr[weight_delta_index].as_uint64(),
                                self.uint_to_bool(delta_hidden_weight_signs_arr[weight_delta_index].as_uint64()),
                                grad_hidden_weight_magnitude,
                                hidden_error_is_negative,
                            )
                            delta_hidden_weight_magnitudes_arr[weight_delta_index] = arc4.UInt64(new_mag)
                            delta_hidden_weight_signs_arr[weight_delta_index] = arc4.UInt64(self.bool_to_uint(new_sign))

                    new_mag, new_sign = self.signed_add(
                        delta_hidden_bias_magnitudes[error_index].as_uint64(),
                        self.uint_to_bool(delta_hidden_bias_signs[error_index].as_uint64()),
                        hidden_error_magnitude,
                        hidden_error_is_negative,
                    )
                    delta_hidden_bias_magnitudes[error_index] = arc4.UInt64(new_mag)
                    delta_hidden_bias_signs[error_index] = arc4.UInt64(self.bool_to_uint(new_sign))

        for i in urange(first_hidden_weight_count):
            delta_first_hidden_weight_magnitudes[i] = arc4.UInt64(
                delta_first_hidden_weight_magnitudes[i].as_uint64() // self.length_data
            )

        if self.hidden_layers > 1:
            for i in urange(hidden_weight_count):
                delta_hidden_weight_magnitudes_arr[i] = arc4.UInt64(
                    delta_hidden_weight_magnitudes_arr[i].as_uint64() // self.length_data
                )

        for i in urange(hidden_state_count):
            delta_hidden_bias_magnitudes[i] = arc4.UInt64(
                delta_hidden_bias_magnitudes[i].as_uint64() // self.length_data
            )

        for i in urange(self.hidden_neurons):
            delta_output_weight_magnitudes[i] = arc4.UInt64(
                delta_output_weight_magnitudes[i].as_uint64() // self.length_data
            )

        delta_output_bias_magnitude = delta_output_bias_magnitude // self.length_data

        for i in urange(first_hidden_weight_count):
            update_amount = self.scale_down_product(
                self.learning_rate,
                delta_first_hidden_weight_magnitudes[i].as_uint64(),
            )
            current_mag = first_hidden_weight_magnitudes[i].as_uint64()
            current_sign = self.uint_to_bool(first_hidden_weight_signs[i].as_uint64())
            delta_sign = self.uint_to_bool(delta_first_hidden_weight_signs[i].as_uint64())

            new_mag, new_sign = self.signed_sub(
                current_mag,
                current_sign,
                update_amount,
                delta_sign,
            )

            first_hidden_weight_magnitudes[i] = arc4.UInt64(new_mag)
            first_hidden_weight_signs[i] = arc4.UInt64(self.bool_to_uint(new_sign))

        if self.hidden_layers > 1:
            for i in urange(hidden_weight_count):
                update_amount = self.scale_down_product(
                    self.learning_rate,
                    delta_hidden_weight_magnitudes_arr[i].as_uint64(),
                )
                current_mag = hidden_weight_magnitudes[i].as_uint64()
                current_sign = self.uint_to_bool(hidden_weight_signs[i].as_uint64())
                delta_sign = self.uint_to_bool(delta_hidden_weight_signs_arr[i].as_uint64())

                new_mag, new_sign = self.signed_sub(
                    current_mag,
                    current_sign,
                    update_amount,
                    delta_sign,
                )

                hidden_weight_magnitudes[i] = arc4.UInt64(new_mag)
                hidden_weight_signs[i] = arc4.UInt64(self.bool_to_uint(new_sign))

        for i in urange(hidden_state_count):
            update_amount = self.scale_down_product(
                self.learning_rate,
                delta_hidden_bias_magnitudes[i].as_uint64(),
            )
            current_mag = hidden_bias_magnitudes[i].as_uint64()
            current_sign = self.uint_to_bool(hidden_bias_signs[i].as_uint64())
            delta_sign = self.uint_to_bool(delta_hidden_bias_signs[i].as_uint64())

            new_mag, new_sign = self.signed_sub(
                current_mag,
                current_sign,
                update_amount,
                delta_sign,
            )

            hidden_bias_magnitudes[i] = arc4.UInt64(new_mag)
            hidden_bias_signs[i] = arc4.UInt64(self.bool_to_uint(new_sign))

        for i in urange(self.hidden_neurons):
            update_amount = self.scale_down_product(
                self.learning_rate,
                delta_output_weight_magnitudes[i].as_uint64(),
            )
            current_mag = output_weight_magnitudes[i].as_uint64()
            current_sign = self.uint_to_bool(output_weight_signs[i].as_uint64())
            delta_sign = self.uint_to_bool(delta_output_weight_signs[i].as_uint64())

            new_mag, new_sign = self.signed_sub(
                current_mag,
                current_sign,
                update_amount,
                delta_sign,
            )

            output_weight_magnitudes[i] = arc4.UInt64(new_mag)
            output_weight_signs[i] = arc4.UInt64(self.bool_to_uint(new_sign))

        output_bias_update_amount = self.scale_down_product(
            self.learning_rate,
            delta_output_bias_magnitude,
        )
        self.output_bias_magnitude, self.output_bias_is_negative = self.signed_sub(
            self.output_bias_magnitude,
            self.output_bias_is_negative,
            output_bias_update_amount,
            delta_output_bias_is_negative,
        )

        self.first_hidden_weight_magnitudes_box.value = first_hidden_weight_magnitudes.copy()
        self.first_hidden_weight_signs_box.value = first_hidden_weight_signs.copy()

        self.hidden_weight_magnitudes_box.value = hidden_weight_magnitudes.copy()
        self.hidden_weight_signs_box.value = hidden_weight_signs.copy()

        self.hidden_bias_magnitudes_box.value = hidden_bias_magnitudes.copy()
        self.hidden_bias_signs_box.value = hidden_bias_signs.copy()

        self.output_weight_magnitudes_box.value = output_weight_magnitudes.copy()
        self.output_weight_signs_box.value = output_weight_signs.copy()

        self.epochs_completed += UInt64(1)

        if self.epochs_completed == self.epochs:
            self.training_complete = True
            self.ready_for_training = False

        self.fees_used += Txn.fee

    @abimethod
    def predict(self, age: UInt64, gender: UInt64) -> UInt64:
        assert self.training_complete == True, "Model has not been fully trained"

        hidden_state_count = self.hidden_layers * self.hidden_neurons

        first_hidden_weight_magnitudes = self.first_hidden_weight_magnitudes_box.value.copy()
        first_hidden_weight_signs = self.first_hidden_weight_signs_box.value.copy()

        hidden_weight_magnitudes = self.hidden_weight_magnitudes_box.value.copy()
        hidden_weight_signs = self.hidden_weight_signs_box.value.copy()

        hidden_bias_magnitudes = self.hidden_bias_magnitudes_box.value.copy()
        hidden_bias_signs = self.hidden_bias_signs_box.value.copy()

        output_weight_magnitudes = self.output_weight_magnitudes_box.value.copy()
        output_weight_signs = self.output_weight_signs_box.value.copy()

        hidden_activations = self.make_zero_array(hidden_state_count)

        for layer in urange(self.hidden_layers):
            for neuron in urange(self.hidden_neurons):
                bias_index = self.hidden_bias_index(layer, neuron)
                preactivation_magnitude = hidden_bias_magnitudes[bias_index].as_uint64()
                preactivation_is_negative = self.uint_to_bool(hidden_bias_signs[bias_index].as_uint64())

                if layer == 0:
                    age_weight_index = self.first_hidden_weight_index(neuron, UInt64(0))
                    gender_weight_index = self.first_hidden_weight_index(neuron, UInt64(1))

                    preactivation_magnitude, preactivation_is_negative = self.accumulate_signed_scaled_product(
                        preactivation_magnitude,
                        preactivation_is_negative,
                        first_hidden_weight_magnitudes[age_weight_index].as_uint64(),
                        self.uint_to_bool(first_hidden_weight_signs[age_weight_index].as_uint64()),
                        age,
                    )

                    preactivation_magnitude, preactivation_is_negative = self.accumulate_signed_scaled_product(
                        preactivation_magnitude,
                        preactivation_is_negative,
                        first_hidden_weight_magnitudes[gender_weight_index].as_uint64(),
                        self.uint_to_bool(first_hidden_weight_signs[gender_weight_index].as_uint64()),
                        gender,
                    )
                else:
                    for prev_neuron in urange(self.hidden_neurons):
                        weight_index = self.hidden_weight_index(layer, neuron, prev_neuron)
                        prev_activation_index = self.hidden_bias_index(layer - UInt64(1), prev_neuron)

                        preactivation_magnitude, preactivation_is_negative = self.accumulate_signed_scaled_product(
                            preactivation_magnitude,
                            preactivation_is_negative,
                            hidden_weight_magnitudes[weight_index].as_uint64(),
                            self.uint_to_bool(hidden_weight_signs[weight_index].as_uint64()),
                            hidden_activations[prev_activation_index].as_uint64(),
                        )

                hidden_activations[bias_index] = arc4.UInt64(
                    self.relu(preactivation_magnitude, preactivation_is_negative)
                )

        z_out_magnitude = self.output_bias_magnitude
        z_out_is_negative = self.output_bias_is_negative

        for neuron in urange(self.hidden_neurons):
            last_hidden_index = self.hidden_bias_index(self.hidden_layers - UInt64(1), neuron)
            z_out_magnitude, z_out_is_negative = self.accumulate_signed_scaled_product(
                z_out_magnitude,
                z_out_is_negative,
                output_weight_magnitudes[neuron].as_uint64(),
                self.uint_to_bool(output_weight_signs[neuron].as_uint64()),
                hidden_activations[last_hidden_index].as_uint64(),
            )

        return self.hard_sigmoid(z_out_magnitude, z_out_is_negative)

    @abimethod
    def clear_data(self) -> None:
        self.age_inputs_box.value = Data()
        self.gender_inputs_box.value = Data()
        self.y_targets_box.value = Data()

        self.first_hidden_weight_magnitudes_box.value = Data()
        self.first_hidden_weight_signs_box.value = Data()

        self.hidden_weight_magnitudes_box.value = Data()
        self.hidden_weight_signs_box.value = Data()

        self.hidden_bias_magnitudes_box.value = Data()
        self.hidden_bias_signs_box.value = Data()

        self.output_weight_magnitudes_box.value = Data()
        self.output_weight_signs_box.value = Data()

        self.output_bias_magnitude = UInt64(0)
        self.output_bias_is_negative = False

        self.hidden_layers = UInt64(0)
        self.hidden_neurons = UInt64(0)

        self.length_data = UInt64(0)
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
    def initialize_parameters(self) -> None:
        first_hidden_weight_count = self.hidden_neurons * self.input_feature_count

        hidden_weight_count = UInt64(0)
        if self.hidden_layers > 1:
            hidden_weight_count = (self.hidden_layers - UInt64(1)) * self.hidden_neurons * self.hidden_neurons

        hidden_bias_count = self.hidden_layers * self.hidden_neurons
        output_weight_count = self.hidden_neurons

        self.write_zero_box(self.first_hidden_weight_magnitudes_box, first_hidden_weight_count)
        self.write_zero_box(self.first_hidden_weight_signs_box, first_hidden_weight_count)

        self.write_zero_box(self.hidden_weight_magnitudes_box, hidden_weight_count)
        self.write_zero_box(self.hidden_weight_signs_box, hidden_weight_count)

        self.write_zero_box(self.hidden_bias_magnitudes_box, hidden_bias_count)
        self.write_zero_box(self.hidden_bias_signs_box, hidden_bias_count)

        self.write_zero_box(self.output_weight_magnitudes_box, output_weight_count)
        self.write_zero_box(self.output_weight_signs_box, output_weight_count)

    @subroutine
    def write_zero_box(self, box_ref: Box[Data], count: UInt64) -> None:
        arr = self.make_zero_array(count)

        if box_ref.get(default=Data()).length == 0:
            box_ref.create(size=arr.bytes.length)
        else:
            box_ref.resize(new_size=arr.bytes.length)

        box_ref.value = arr.copy()

    @subroutine
    def first_hidden_weight_index(self, neuron: UInt64, feature: UInt64) -> UInt64:
        return neuron * self.input_feature_count + feature

    @subroutine
    def hidden_weight_index(self, layer: UInt64, neuron: UInt64, prev_neuron: UInt64) -> UInt64:
        return ((layer - UInt64(1)) * self.hidden_neurons * self.hidden_neurons) + (neuron * self.hidden_neurons) + prev_neuron

    @subroutine
    def hidden_bias_index(self, layer: UInt64, neuron: UInt64) -> UInt64:
        return (layer * self.hidden_neurons) + neuron

    @subroutine
    def make_zero_array(self, count: UInt64) -> Data:
        arr = Data()
        for epoch in urange(count):
            arr.append(arc4.UInt64(0))
        return arr.copy()

    @subroutine
    def zero_existing_array(self, arr: Data, count: UInt64) -> None:
        for i in urange(count):
            arr[i] = arc4.UInt64(0)

    @subroutine
    def scale_down_product(self, a: UInt64, b: UInt64) -> UInt64:
        return op.btoi(((BigUInt(a) * b) // self.scale_factor).bytes)

    @subroutine
    def accumulate_signed_scaled_product(
        self,
        acc_magnitude: UInt64,
        acc_is_negative: bool,
        weight_magnitude: UInt64,
        weight_is_negative: bool,
        unsigned_input: UInt64,
    ) -> tuple[UInt64, bool]:
        product_magnitude = self.scale_down_product(weight_magnitude, unsigned_input)
        return self.signed_add(
            acc_magnitude,
            acc_is_negative,
            product_magnitude,
            weight_is_negative,
        )

    @subroutine
    def relu(self, magnitude: UInt64, is_negative: bool) -> UInt64:
        if is_negative:
            return UInt64(0)
        return magnitude

    @subroutine
    def relu_derivative(self, magnitude: UInt64, is_negative: bool) -> UInt64:
        if is_negative:
            return UInt64(0)
        if magnitude == 0:
            return UInt64(0)
        return UInt64(1)

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
    def xor_signs(self, a: bool, b: bool) -> bool:
        if a == b:
            return False
        return True

    @subroutine
    def bool_to_uint(self, value: bool) -> UInt64:
        if value:
            return UInt64(1)
        return UInt64(0)

    @subroutine
    def uint_to_bool(self, value: UInt64) -> bool:
        if value == 0:
            return False
        return True

    @subroutine
    def signed_add(
        self,
        a_magnitude: UInt64,
        a_is_negative: bool,
        b_magnitude: UInt64,
        b_is_negative: bool,
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
        b_is_negative: bool,
    ) -> tuple[UInt64, bool]:
        return self.signed_add(
            a_magnitude,
            a_is_negative,
            b_magnitude,
            not b_is_negative,
        )

    @abimethod
    def dummy(self) -> None:
        return