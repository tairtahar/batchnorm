import models_handling
import utils
import visualizations
import argparse
import pickle


def main():
    """main function is in charge of training the different networks tested and compared. The evaluation of the
    performance is done over mnist database.
    The networks are based on LeNet architecture and they differ by the batch normalization layers: In the first
    network tested there is no batch normalization layer, the second network has batch normalization on the first layer,
     the third has batch normalization on the first two layers and so on until the last network that has batch
     normalization in all the layers.
    There are several parameters that are set to default values, but they can be changed by the user when execution the
    program. More information regarding them is presented when using help.
    batch_size - batch size for training
    output_size - in the case of mnist should be 10
    epochs - number of epochs for the training
    optimizer - optimizer for the training, default 'adam'
    window_size - In inference time batch norm standardization is performed over moving window of window_size (number)
    of epochs
    verbose - whether to print the training details
    flag_visualization - 1 in case that we want to visualize the results on a graph
    """

    # DATA LOADING AND PREPROCESSING
    data = utils.data_prepare()

    # PARAMETERS INITIALIZATION.
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size for training", default=64)
    parser.add_argument("--output_size", help="output size, number of classes", default=10)  # mnist has 10 possible classes
    parser.add_argument("--epochs", help="number of epochs for training", default=10)
    parser.add_argument("--optimizer", help="optimizer for training", default='adam')
    parser.add_argument("--epsilon", help="number of epochs for moving standardization during inference time",
                        default=0.00000001)
    parser.add_argument("--window_size", help="window size for averaging in batchnorm algorithm", default=5)
    parser.add_argument("--verbose", help="verbosity", default=1)
    parser.add_argument("--flag_visualizations", help="flag for presenting plots of the training process", default=1)

    args = parser.parse_args()

    args_dict = vars(args)
    with open('temp_data/arguments', 'wb') as file_pi:
        pickle.dump(args_dict, file_pi)

    '''It is possible to choose (copy to the next line) one of the following:
    (1)'lenet' (no BN) ; (2)'lenet_bn1' (first conv layer has BN) ; (3)'lenet_bn2' (first+second conv layers have BN);
    (4)lenet_fc_bn1 (conv+first FC layer have BN) ; (5)'lenet_fc_bn2' (all layers with BN);'''
    networks = ['lenet', 'lenet_bn1', 'lenet_bn2', 'lenet_fc_bn1', 'lenet_fc_bn2']
    histories = []
    for i in range(len(networks)):
        network = networks[i]
        # MODEL CREATION AND TRAINING
        history = models_handling.model_execution(network=network,
                                                  data=data,
                                                  output_size=args.output_size,
                                                  batch_size=args.batch_size,
                                                  optimizer=args.optimizer,
                                                  epochs=args.epochs,
                                                  epsilon=args.epsilon,
                                                  window_size=args.window_size,
                                                  verbose=args.verbose)
        histories.append(list(history.history['accuracy']))
        # VISUALIZATION OF THE TRAINING PROCESS
    if args.flag_visualizations:
        visualizations.plot_accuracies(histories, networks)

    with open('temp_data/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    main()

