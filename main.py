import tensorflow as tf

import models
import utils
import visualizations

# Data loading and preprocessing
x_train, x_val, x_test, y_val, y_train, y_test = utils.data_prepare()

# Parameters definition. Please adjust and choose a network
input_shape = x_train.shape[1:]
batch_size = 256
epochs = 15
output_size = 10
optimizer = 'sgd'
verbose = 1
flag_plot_accuracy = 1
flag_plot_loss = 1
network = 'lenet'  # 'lenet_bn1', 'lenet_bn2', 'lenet_fc_bn1'

# Model creation and training
lenet_model = models.model_selection(network, input_shape, output_size, batch_size)
lenet_model.model_compilation(optimizer=optimizer)
history = lenet_model.train(x_train, y_train, x_val, y_val,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose)
lenet_model.evaluate(x_test, y_test, verbose=verbose)

# Visualizations of the training process
visualizations.plot_accuracy(history)
visualizations.plot_loss(history)
