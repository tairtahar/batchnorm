import models_handling
import utils
import visualizations


def main():
    # Data loading and preprocessing
    data = utils.data_prepare()

    # Parameters definition. Please adjust params and choose a network
    batch_size = 64
    epochs = 10
    output_size = 10  # mnist has 10 possible classes
    optimizer = 'adam'
    window_size = 5  # for the moving average in the last part of batchnorm algorithm
    verbose = 1
    flag_visualizations = 1
    '''You cab choose (copy to the next line) one of the following:
    (1)'lenet' (no BN) ; (2)'lenet_bn1' (first conv layer has BN) ; (3)'lenet_bn2' (first+second conv layers have BN);
    (4)lenet_fc_bn1 (conv+first FC layer have BN) ; (5)'lenet_fc_bn2' (all layers with BN);'''
    networks = ['lenet', 'lenet_bn1' , 'lenet_bn2', 'lenet_fc_bn1', 'lenet_fc_bn2']
    histories = []
    for i in range(len(networks)):
        network = networks[i]
        # Model creation and training
        history = models_handling.model_execution(network=network,
                                                  data=data,
                                                  output_size=output_size,
                                                  batch_size=batch_size,
                                                  optimizer=optimizer,
                                                  epochs=epochs,
                                                  window_size=window_size,
                                                  verbose=verbose)
        histories.append(history)
        # Visualizations of the training process
    if flag_visualizations:
        visualizations.plot_accuracies(histories, networks)
        # visualizations.plot_loss(history)


if __name__ == '__main__':
    main()

