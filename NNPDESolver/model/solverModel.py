import tensorflow as tf
from domain_conditions import Domain_Condition
from boundary_conditions import Boundary_Condition
from tensorflow.keras import initializers


class Solver_Model():
    def __init__(self):

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
        self._metrics = tf.keras.metrics.Mean(0.001)

        initializer = tf.keras.initializers.GlorotNormal()

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
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer6')(x)
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer7')(x)
        x = tf.keras.layers.Dense(
            units=20, activation=tf.nn.sigmoid, name='layer8')(x)
        outputs = tf.keras.layers.Dense(units=1, name='output')(x)

        self._model = tf.keras.Model(inputs=inputs, outputs=outputs)

        self._initial_conditions = []
        self._boundary_conditions = []
        self._domain_condition = None

    def set_domain_condition(self, condition: Domain_Condition):
        self._domain_condition = condition

    def add_initial_condition(self, condition):
        self._initial_conditions.append(condition)

    def add_boundary_condition(self, condition: Boundary_Condition):
        self._boundary_conditions.append(condition)

    def predict(self, x_input, t_input):
        tf.print("Start prediction")
        x_data, t_data = tf.meshgrid(x_input, t_input)
        x_data = tf.reshape(x_data, [-1])
        t_data = tf.reshape(t_data, [-1])
        input_data = tf.stack([x_data, t_data], -1)

        return self._model(input_data)

    def train(self, epochs):
        tf.print("Start training")

        self.losses_number = len(self._initial_conditions) + \
            len(self._boundary_conditions)

        if(self._domain_condition != None):
            self.losses_number = self.losses_number+1

        self._metrics.reset_states()

        for epoch in range(epochs):
            self.train_epoch()

            if(epoch % 500 == 0):
                print("Epoch {0}".format(epoch))

    @tf.function
    def train_epoch(self):
        with tf.GradientTape() as tape:

            if(self._domain_condition != None):
                loss_total = tf.cast(self._domain_condition.computeLoss(
                    self._model, self._optimizer)/self.losses_number, tf.float32)
                # tf.print("domain: ", loss_total)
            else:
                loss_total = tf.cast(0.0, tf.float32)

            for condition in self._initial_conditions:
                loss = tf.cast(condition.computeLoss(
                    self._model, self._optimizer)/self.losses_number, tf.float32)
                # tf.print("initial: ", loss)
                loss_total = loss_total + loss

            for condition in self._boundary_conditions:
                loss = tf.cast(condition.computeLoss(
                    self._model, self._optimizer)/self.losses_number, tf.float32)
                # tf.print("boundary: ", loss)
                loss_total = loss_total + loss

            tf.print("total: ", loss_total)

        trainable_vars = self._model.trainable_variables
        gradients = tape.gradient(loss_total, trainable_vars)
        self._optimizer.apply_gradients(zip(gradients, trainable_vars))
