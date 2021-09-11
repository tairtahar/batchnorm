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
                                activation='relu')(self.input1)
        self.s2 = layers.AveragePooling2D(padding='valid')(self.c1)
        self.c3 = layers.Conv2D(filters=16,
                                kernel_size=(3, 3),
                                padding='valid',
                                activation='relu')(self.s2)
        self.s4 = layers.AveragePooling2D(padding='valid')(self.c3)
        self.flatten = layers.Flatten()(self.s4)
        self.c5 = layers.Dense(units=120,
                               activation='relu')(self.flatten)
        self.f6 = layers.Dense(
            units=84,
            activation='relu')(self.c5)
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
    def __init__(self, input_shape, output_size=10):
        super().__init__(input_shape)
        # self.norm1 = layers.Lambda(batch_norm)(self.c1)
        self.affine1 = BatchNormLayer(input_shape)(self.c1)
        self.s2 = layers.AveragePooling2D(padding='valid')(self.affine1)
        # self.bn2 = layers.Lambda(batch_norm)(self.c3)
        # self.s4 = layers.AveragePooling2D(padding='valid')(self.bn2)
        self.model = models.Model(inputs=self.input1, outputs=self.output_layer)


class BatchNormLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super().__init__()
        # self.units = units
        gamma_init = tf.random_normal_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=(input_shape[1:]),
                                     dtype='float32'),
            trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=(input_shape[1:]), dtype='float32'),
            trainable=True)

    # def build(self, input_shape):  # Create the state of the layer (weights)
    # gamma_init = tf.random_normal_initializer()
    # self.gamma = tf.Variable(
    #     initial_value=gamma_init(shape=(input_shape[1:]),
    #                              dtype='float32'),
    #     trainable=True)
    # beta_init = tf.zeros_initializer()
    # self.beta = tf.Variable(
    #     initial_value=beta_init(shape=(input_shape[1:]), dtype='float32'),
    #     trainable=True)

    def call(self, inputs):  # , training=None):  # Defines the computation from inputs to outputs
        # if training:
        epsilon = 0.00000001
        mu = K.mean(inputs, axis=0)
        variance = K.var(inputs)
        x_hat = (inputs - mu) / (variance + epsilon)
        # gamma_rep = k.expand_dims(self.gamma, axis=1)
        # beta_rep = k.expand_dims(self.beta, axis=1)
        # gamma_rep = k.repeat_elements(gamma_rep, inputs.shape[0], 1)
        # beta_rep = k.repeat_elements(beta_rep, inputs.shape[0], 1)
        # # for i in range(1,inputs.shape()[0]): # replace inputs with x_hat
        # outputs = x_hat * gamma_rep + beta_rep
        return x_hat  # outputs
