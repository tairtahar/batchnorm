import pickle
import models_handling
import utils
import visualizations
from types import SimpleNamespace


def model_baseline(network):
    data = utils.data_prepare()
    args_saved = pickle.load(open('temp_data/arguments', "rb"))
    args = SimpleNamespace(**args_saved)
    history = models_handling.model_execution(network=network,
                                              data=data,
                                              output_size=args.output_size,
                                              batch_size=args.batch_size,
                                              optimizer=args.optimizer,
                                              epochs=args.epochs,
                                              window_size=args.window_size,
                                              verbose=args.verbose)

    return history.history['accuracy']


def comparison_keras_batchnorm():
    history = pickle.load(open('temp_data/trainHistoryDict', "rb"))['accuracy']
    histories = list()
    histories.append(history)
    network = 'keras'
    networks = ['Manual', network]
    keras_training = model_baseline(network)
    histories.append(keras_training)
    visualizations.plot_accuracies(histories, networks)


comparison_keras_batchnorm()
