import models_handling
import utils
import visualizations
import argparse


def main():
    # Data loading and preprocessing
    data = utils.data_prepare()

    # Parameters definition. Please adjust params and choose a network
    # batch_size = 64
    # epochs = 10
    output_size = 10  # mnist has 10 possible classes
    # optimizer = 'adam'
    # window_size = 5  # for the moving average in the last part of batchnorm algorithm
    # verbose = 1
    # flag_visualizations = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size", default=64)
    parser.add_argument("--epochs", help="number of epochs", default=10)
    # parser.add_argument("output_size", help="echo the string you use here", default=10)
    parser.add_argument("--optimizer", help="optimizer", default='adam')
    parser.add_argument("--window_size", help="window size for averaging in batchnorm algorithm", default=5)
    parser.add_argument("--verbose", help="verbosity", default=1)
    parser.add_argument("--flag_visualizations", help="plot flag", default=1)

    args = parser.parse_args()

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
                                                  output_size=output_size,
                                                  batch_size=args.batch_size,
                                                  optimizer=args.optimizer,
                                                  epochs=args.epochs,
                                                  window_size=args.window_size,
                                                  verbose=args.verbose)
        histories.append(history)
        # Visualizations of the training process
    if args.flag_visualizations:
        visualizations.plot_accuracies(histories, networks)
        # visualizations.plot_loss(history)


if __name__ == '__main__':
    main()

