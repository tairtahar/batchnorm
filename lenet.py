import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


class LeNet(tf.keras.Model):
    def __init__(self, input_shape, output_size=10):
        super(LeNet, self).__init__()
        if input_shape is None:
            input_shape = (32, 32, 1)
        self.input1 = layers.Input(shape=input_shape)
        self.c1 = layers.Conv2D(filters=6,
                                input_shape=input_shape,
                                kernel_size=(5, 5),
                                padding='valid',
                                activation='sigmoid'
                                )
        self.s2 = layers.AveragePooling2D(padding='valid')
        self.c3 = layers.Conv2D(filters=16,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='sigmoid')
        self.s4 = layers.AveragePooling2D(padding='valid')
        self.flatten = layers.Flatten()
        self.c5 = layers.Dense(units=120,
                               activation='sigmoid')
        self.f6 = layers.Dense(
            units=84,
            activation='sigmoid')
        self.output_layer = layers.Dense(
            units=output_size,
            activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.c1(inputs)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.f6(x)
        return self.output_layer(x)

    def model_compilation(self, optimizer):
        self.compile(optimizer=optimizer,
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

        print("compilation done")

    def train(self, x_train, y_train, x_val, y_val, batch_size=128, epochs=5, verbose=0):
        history = self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(x_val, y_val),
                           verbose=verbose,
                           )
        print("model training done")
        return history

    def evaluation(self, x_test, y_test, verbose=0):
        score = self.evaluate(x_test, y_test, verbose=0)
        print('test loss:', score[0])
        print('test accuracy:', score[1])


class LeNetBN1(LeNet):
    def __init__(self, input_shape, output_size=10):
        super().__init__(input_shape)
        self.c1 = layers.Conv2D(filters=6,
                                input_shape=input_shape,
                                kernel_size=(5, 5),
                                padding='valid',
                                )  # no activation
        self.affine1 = BatchNormLayer([28, 28, 6])

    def call(self, inputs, training=False):
        # if training:
        x = self.c1(inputs)
        x = self.affine1(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.f6(x)
        return self.output_layer(x)

        # self.input1 = layers.Input(shape=input_shape)
        # self.c1 = layers.Conv2D(filters=6,
        #                         input_shape=input_shape,
        #                         kernel_size=(5, 5),
        #                         padding='valid',
        #                         )(self.input1)  # no activation
        # self.s2 = layers.AveragePooling2D(padding='valid')(tf.keras.activations.sigmoid(self.affine1(self.c1)))
        # self.c3 = layers.Conv2D(filters=16,
        #                         kernel_size=(3, 3),
        #                         padding='valid',
        #                         activation='sigmoid')(self.s2)
        # self.s4 = layers.AveragePooling2D(padding='valid')(self.c3)
        # self.flatten = layers.Flatten()(self.s4)
        # self.c5 = layers.Dense(units=120,
        #                        activation='sigmoid')(self.flatten)
        # self.f6 = layers.Dense(
        #     units=84,
        #     activation='sigmoid')(self.c5)
        # self.output_layer = layers.Dense(
        #     units=output_size,
        #     activation=tf.nn.softmax)(self.f6)
        # self.model = models.Model(inputs=self.input1, outputs=self.output_layer)


class LeNetBN2(LeNetBN1):
    def __init__(self, input_shape, batch_size, output_size=10):
        super().__init__(input_shape, batch_size)
        self.c3 = layers.Conv2D(filters=16,
                                kernel_size=(3, 3),
                                padding='valid')(self.s2)  # no activation
        self.affine2 = BatchNormLayer(self.c3.shape[1:], batch_size)(self.c3)
        self.activated2 = tf.keras.activations.sigmoid(self.affine2)
        self.s4 = layers.AveragePooling2D(padding='valid')(self.activated2)
        self.model = models.Model(inputs=self.input1, outputs=self.output_layer)


class LeNetFCBN1(LeNet):
    def __init__(self, input_shape, batch_size, output_size=10):
        super().__init__(input_shape)
        self.c5 = layers.Dense(units=120)(self.flatten)
        self.affine3 = BatchNormFCLayer(batch_size)(self.c5)
        self.activated3 = tf.keras.activations.sigmoid(self.affine3)
        self.f6 = layers.Dense(units=84, activation='relu')(self.activated3)
        # self.model = models.Model(inputs=self.input1, outputs=self.output_layer)


class BatchNormLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super().__init__()

        # def build(self, input_shape):
        gamma_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=input_shape, dtype='float32'), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=input_shape, dtype='float32'), trainable=True)

    def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
        # if training:
        epsilon = 0.00000001
        mu = K.mean(inputs, axis=(0, 1, 2), keepdims=True)
        variance = K.var(inputs, axis=(0, 1, 2), keepdims=True)
        x_hat = (inputs - mu) / K.sqrt(variance + epsilon)
        outputs = self.gamma * x_hat + self.beta

        return outputs


class BatchNormFCLayer(tf.keras.layers.Layer):  # for the case of fully connected (1D inputs)
    def __init__(self):
        super().__init__()
        # self.units = units
        gamma_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=[1], dtype='float32'), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=[1], dtype='float32'), trainable=True)

    def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
        epsilon = 0.00000001
        mu = K.mean(inputs, axis=0)
        variance = K.var(inputs, axis=0)
        x_hat = (inputs - mu) / K.sqrt(variance + epsilon)
        outputs = self.gamma * x_hat + self.beta

        return outputs
