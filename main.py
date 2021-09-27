import models_handling
import utils
import visualizations
import argparse
import pickle


def main():
    # Data loading and preprocessing
    data = utils.data_prepare()

    # Parameters definition. Please adjust params and choose a network
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", default=64)
    parser.add_argument("--output_size", help="output size", default=10)  # mnist has 10 possible classes
    parser.add_argument("--epochs", help="number of epochs", default=10)
    parser.add_argument("--optimizer", help="optimizer", default='adam')
    parser.add_argument("--window_size", help="window size for averaging in batchnorm algorithm", default=5)
    parser.add_argument("--verbose", help="verbosity", default=1)
    parser.add_argument("--flag_visualizations", help="plot flag", default=1)

    args = parser.parse_args()

    args_dict = vars(args)
    with open('arguments', 'wb') as file_pi:
        pickle.dump(args_dict, file_pi)

    '''You cab choose (copy to the next line) one of the following:
    (1)'lenet' (no BN) ; (2)'lenet_bn1' (first conv layer has BN) ; (3)'lenet_bn2' (first+second conv layers have BN);
    (4)lenet_fc_bn1 (conv+first FC layer have BN) ; (5)'lenet_fc_bn2' (all layers with BN);'''
    networks = ['lenet', 'lenet_bn1', 'lenet_bn2', 'lenet_fc_bn1', 'lenet_fc_bn2']
    histories = []
    for i in range(len(networks)):
        network = networks[i]
        # Model creation and training
        history = models_handling.model_execution(network=network,
                                                  data=data,
                                                  output_size=args.output_size,
                                                  batch_size=args.batch_size,
                                                  optimizer=args.optimizer,
                                                  epochs=args.epochs,
                                                  window_size=args.window_size,
                                                  verbose=args.verbose)
        histories.append(history.history)
        # Visualizations of the training process
    if args.flag_visualizations:
        visualizations.plot_accuracies(histories, networks)

    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    main()

