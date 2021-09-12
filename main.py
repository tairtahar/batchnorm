from lenet2 import LeNet, LeNetBN1, LeNetBN2
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# shuffling the data, making sure the division is not biased

x = tf.concat([x_train, x_test], 0)
y = tf.concat([y_train, y_test], 0)
indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
tf.random.set_seed(123)
shuffled_indices = tf.random.shuffle(indices)
x = tf.gather(x, shuffled_indices)
y = tf.gather(y, shuffled_indices)
x_train = x[:-10000, :, :]
x_test = x[-10000:, :, :]
y_train = y[:-10000]
y_test = y[-10000:]

###############
x_train = tf.pad(x_train, [[0, 0], [2, 2], [2, 2]]) / 255
x_test = tf.pad(x_test, [[0, 0], [2, 2], [2, 2]]) / 255
indices = tf.range(start=0, limit=tf.shape(x_train)[0], dtype=tf.int32)
tf.random.set_seed(1234)
shuffled_indices = tf.random.shuffle(indices)
x_train = tf.gather(x_train, shuffled_indices)
y_train = tf.gather(y_train, shuffled_indices)
x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)
x_val = x_train[-2000:, :, :, :]
y_val = y_train[-2000:]
x_train = x_train[:-2000, :, :, :]
y_train = y_train[:-2000]
input_shape = x_train.shape[1:]
# lenet_model = LeNet(input_shape=input_shape, output_size=10)
batch_size = 256
lenet_model = LeNetBN2(input_shape=input_shape, batch_size=batch_size, output_size=10)
lenet_model.model_compilation(optimizer='adam')
history = lenet_model.train(x_train, y_train, x_val, y_val, batch_size=batch_size, epochs=15,
                            verbose=1)  # steps_per_epoch=200,
lenet_model.evaluate(x_test, y_test, verbose=1)

# Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('accuracy.pdf')
plt.show()
# Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
# plt.savefig('loss.pdf')
plt.show()
