import tensorflow as tf


def load_data():
    """ This function loads the data from tensorflow datasets """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x = tf.concat([x_train, x_test], 0)
    y = tf.concat([y_train, y_test], 0)

    return x, y


def shuffle_data(x, y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    tf.random.set_seed(123)
    shuffled_indices = tf.random.shuffle(indices)
    shuffled_x = tf.gather(x, shuffled_indices)
    shuffled_y = tf.gather(y, shuffled_indices)

    return shuffled_x, shuffled_y


def divide_data(x, y, idx):
    x_part1 = x[:idx, :, :]
    x_part2 = x[idx:, :, :]
    y_part1 = y[:idx]
    y_part2 = y[idx:]

    return x_part1, x_part2, y_part1, y_part2


def zero_padding_and_norm(data):
    """ mnist images have size 28x28 but input layer in LeNet has 32x32"""
    return tf.pad(data, [[0, 0], [2, 2], [2, 2]]) / 255


def data_prepare():
    """ This is the main function of loading the data and perform preprocessing. The concatenation of train and test
    data is performed since I noticed that test accuracy was higher for some cases, so avoiding 'too easy test data'
    I will divide them data manually after shuffling it in the next steps """
    x, y = load_data()
    x, y = shuffle_data(x, y)
    x_train, x_test, y_train, y_test = divide_data(x, y, -10000)  # test dataset has 10k images
    # (same as the original size)
    x_train = zero_padding_and_norm(x_train)
    x_test = zero_padding_and_norm(x_test)
    x_train, x_val, y_train, y_val = divide_data(x_train, y_train, -9000)  # validation dataset has 9k images
    x_train = tf.expand_dims(x_train, axis=3, name=None)
    x_test = tf.expand_dims(x_test, axis=3, name=None)
    x_val = tf.expand_dims(x_val, axis=3, name=None)
    data = x_train, x_val, x_test, y_val, y_train, y_test
    return data


