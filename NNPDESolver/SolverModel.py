import tensorflow as tf


class PDE_Solver():
    def __init__(self, domain_loss_calculator):

        self._domain_loss_calculator = domain_loss_calculator

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._metrics = tf.keras.metrics.Mean(0.001)

        inputs = tf.keras.Input(shape=(2,), name="inputs")
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer1')(inputs)
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer2')(x)
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer3')(x)
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer4')(x)
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer5')(x)
        outputs = tf.keras.layers.Dense(units=1, name='output')(x)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self.initial_conditions = []
        self.boundary_conditions = []

    def add_initial_condition(self, condition):
        self.initial_conditions.append(condition)

    def add_boundary_condition(self, condition):
        self.boundary_conditions.append(condition)

    def predict(self, x_data, t_data):
        print("Start prediction")
        input_data = tf.stack(
            [tf.reshape(x_data, [-1]), tf.reshape(t_data, [-1])], axis=1)
        return self._model(input_data)

    def train(self, x_data, t_data, epochs):
        print("Start training")
        self._metrics.reset_states()

        for epoch in range(epochs):
            self.train_epoch(x_data, t_data)

            if(epoch % 500 == 0):
                print("Epoch {0}, Loss {1}".format(
                    epoch, self._metrics.result()))

    @tf.function
    def train_epoch(self, x_data, t_data):
        for condition in self.initial_conditions:
            condition.train(self._model, self._optimizer)

        for condition in self.boundary_conditions:
            condition.train(self._model, self._optimizer)

        with tf.GradientTape() as tape:
            x_data_reshape = tf.reshape(x_data, [-1])
            t_data_reshape = tf.reshape(t_data, [-1])
            input_data = tf.stack([x_data_reshape, t_data_reshape], axis=1)

            # Run forward
            phi = self._model(input_data)
            # Compute the loss value
            loss = self._domain_loss_calculator.loss(
                x_data_reshape, t_data_reshape, phi)

        # Update metrics
        self._metrics.update_state(loss)

        # Compute gradients
        trainable_vars = self._model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self._optimizer.apply_gradients(zip(gradients, trainable_vars))
