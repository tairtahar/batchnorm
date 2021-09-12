import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense
import tensorflow as tf
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
                                activation='sigmoid')(self.input1)
        self.s2 = layers.AveragePooling2D(padding='valid')(self.c1)
        self.c3 = layers.Conv2D(filters=16,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='sigmoid')(self.s2)
        self.s4 = layers.AveragePooling2D(padding='valid')(self.c3)
        self.flatten = layers.Flatten()(self.s4)
        self.c5 = layers.Dense(units=120,
                               activation='sigmoid')(self.flatten)
        self.f6 = layers.Dense(
            units=84,
            activation='sigmoid')(self.c5)
        self.output_layer = layers.Dense(
            units=output_size,
            activation=tf.nn.softmax)(self.f6)
        self.model = models.Model(inputs=self.input1, outputs=self.output_layer)

    def model_compilation(self, optimizer):
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("compilation done")

    def train(self, x_train, y_train, x_val, y_val, batch_size=128, epochs=5, verbose=0):
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                                 validation_data=(x_val, y_val),
                                 verbose=verbose,
                                 )
        print("model training done")
        return history

    def evaluate(self, x_test, y_test, verbose=0):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('test loss:', score[0])
        print('test accuracy:', score[1])


class LeNetBN1(LeNet):
    def __init__(self, input_shape, batch_size, output_size=10):
        super().__init__(input_shape)
        # self.norm1 = layers.Lambda(batch_norm)(self.c1)
        self.affine1 = BatchNormLayer(self.c1.shape[1:], batch_size)(self.c1)
        self.s2 = layers.AveragePooling2D(padding='valid')(self.affine1)
        # self.bn2 = layers.Lambda(batch_norm)(self.c3)
        # self.s4 = layers.AveragePooling2D(padding='valid')(self.bn2)
        self.model = models.Model(inputs=self.input1, outputs=self.output_layer)


class LeNetBN2(LeNet):
    def __init__(self, input_shape, batch_size, output_size=10):
        super().__init__(input_shape)
        # self.norm1 = layers.Lambda(batch_norm)(self.c1)
        self.affine1 = BatchNormLayer(self.c1.shape[1:], batch_size)(self.c1)
        self.s2 = layers.AveragePooling2D(padding='valid')(self.affine1)
        self.affine2 = BatchNormLayer(self.c3.shape[1:], batch_size)(self.c3)
        self.s4 = layers.AveragePooling2D(padding='valid')(self.affine2)
        self.model = models.Model(inputs=self.input1, outputs=self.output_layer)


class BatchNormLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, batch_size):
        super().__init__()
        # self.units = units
        gamma_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=input_shape, dtype='float32'), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=input_shape, dtype='float32'), trainable=True)
        self.batch_size = batch_size

    def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
        epsilon = 0.00000001
        mu = K.mean(inputs, axis=(0, 1, 2), keepdims=True)
        variance = K.var(inputs, axis=(0, 1, 2), keepdims=True)
        x_hat = (inputs - mu) / K.sqrt(variance + epsilon)
        outputs = self.gamma * x_hat + self.beta

        return outputs
