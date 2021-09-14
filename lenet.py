import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow.keras.backend as K


class LeNet(tf.keras.Model):
    """Original LeNet network implementation"""
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
        score = self.evaluate(x_test, y_test, verbose=verbose)
        print('test loss:', score[0])
        print('test accuracy:', score[1])


class LeNetBN1(LeNet):
    """Extends LeNet with batch normalization on the first convolutional layer"""
    def __init__(self, input_shape, output_size, window):
        super().__init__(input_shape, output_size)
        self.c1 = layers.Conv2D(filters=6,
                                input_shape=input_shape,
                                kernel_size=(5, 5),
                                padding='valid',
                                )  # no activation
        self.affine1 = BatchNormLayer([28, 28, 6], window)

    def call(self, inputs):
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


class LeNetBN2(LeNetBN1):
    """Extends LeNetBN1 adding batch normalization also on the second convolutional layer"""
    def __init__(self, input_shape, output_size, window):
        super().__init__(input_shape, output_size, window)
        self.c3 = layers.Conv2D(filters=16,
                                kernel_size=(3, 3),
                                padding='valid')  # no activation
        self.affine2 = BatchNormLayer([12, 12, 16], window)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.affine1(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.affine2(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.f6(x)
        return self.output_layer(x)


class BatchNormLayer(tf.keras.layers.Layer):
    """implementation of the batch normalization layer for convolutional case"""
    def __init__(self, input_shape, window):
        super().__init__()
        gamma_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=input_shape, dtype='float32'), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=input_shape, dtype='float32'), trainable=True)
        self.window = window
        self.moving_mean = []
        self.moving_variance = []

    def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
        mu = K.mean(inputs, axis=(0, 1, 2), keepdims=True)
        variance = K.var(inputs, axis=(0, 1, 2), keepdims=True)
        epsilon = 0.00000001
        outputs = batchnorm_calculations(self, mu, variance, inputs, epsilon, training)
        return outputs


class LeNetFCBN1(LeNetBN2):
    """LeNet with batchnorm on 2 conv layers and first fully connected layer"""
    def __init__(self, input_shape, output_size, window):
        super().__init__(input_shape, output_size, window)
        self.c5 = layers.Dense(units=120)  # No activation
        self.affine3 = BatchNormFCLayer(window)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.affine1(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.affine2(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.affine3(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.f6(x)
        return self.output_layer(x)


class LeNetFCBN2(LeNetFCBN1):
    """LeNet with batchnorm on 2 conv layers and 2 fully connected layers"""
    def __init__(self, input_shape, output_size, window):
        super().__init__(input_shape, output_size, window)
        self.f6 = layers.Dense(units=84)  # No activation
        self.affine4 = BatchNormFCLayer(window)

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.affine1(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.affine2(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.s4(x)
        x = self.flatten(x)
        x = self.c5(x)
        x = self.affine3(x)
        x = tf.keras.activations.sigmoid(x)
        x = self.f6(x)
        x = self.affine4(x)
        x = tf.keras.activations.sigmoid(x)
        return self.output_layer(x)


class BatchNormFCLayer(tf.keras.layers.Layer):  # for the case of fully connected (1D inputs)
    def __init__(self, window):
        super().__init__()
        gamma_init = tf.ones_initializer()
        self.gamma = tf.Variable(
            initial_value=gamma_init(shape=[1], dtype='float32'), trainable=True)
        beta_init = tf.zeros_initializer()
        self.beta = tf.Variable(
            initial_value=beta_init(shape=[1], dtype='float32'), trainable=True)
        self.window = window
        self.moving_mean = []
        self.moving_variance = []

    def call(self, inputs, training=None):  # Defines the computation from inputs to outputs
        epsilon = 0.00000001
        mu = K.mean(inputs, axis=0)
        variance = K.var(inputs, axis=0)
        outputs = batchnorm_calculations(self, mu, variance, inputs, epsilon, training)
        return outputs


def batchnorm_calculations(curr_layer, mu, variance, inputs, epsilon, training):
    """This function contains the needed calculations for the the inference model"""
    if training:  # In case of training, perform batch normalization to learn beta and gamma
        x_hat = (inputs - mu) / K.sqrt(variance + epsilon)
        outputs = curr_layer.gamma * x_hat + curr_layer.beta
    else:  # In case of testing - calculation of the inference model
        curr_layer.moving_mean.append(mu)
        curr_layer.moving_variance.append(variance)
        if len(curr_layer.moving_mean) > curr_layer.window:  # keep the scope of the window
            curr_layer.moving_mean = curr_layer.moving_mean[1:]
            curr_layer.moving_variance = curr_layer.moving_variance[1:]
        if len(curr_layer.moving_mean) == 1:  # In case this is the first batch, than the average will be itself
            current_mean_means = mu
            current_mean_variances = variance
        else:  # In the regular case we calculate the mean and variance according to the given in the paper
            current_mean_means = tf.keras.layers.Average()(curr_layer.moving_mean)
            current_mean_variances = inputs.shape[0] / (inputs.shape[0] - 1) * K.mean(curr_layer.moving_variance)
        outputs = curr_layer.gamma / K.sqrt(current_mean_variances + epsilon) * inputs + \
                  (curr_layer.beta - curr_layer.gamma * current_mean_means / K.sqrt(current_mean_variances + epsilon))
    return outputs
