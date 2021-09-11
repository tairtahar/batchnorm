from lenet2 import LeNet
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
input_shape = x_train.shape[1:]
print(x_train.shape)
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
x_val = x_train[-2000:, :, :, :]
y_val = y_train[-2000:]
x_train = x_train[:-2000, :, :, :]
y_train = y_train[:-2000]
input_shape = x_train.get_shape().as_list()[1:]
lenet_model = LeNet(input_shape=input_shape, output_size=10)
lenet_model.model_compilation(optimizer='adam')
lenet_model.train(x_train, y_train, x_val, y_val, batch_size=256, epochs=5, verbose=1)
lenet_model.evaluate(x_test, y_test, verbose=1)
