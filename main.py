import models_handling
import utils
import visualizations


def main():
    # Data loading and preprocessing
    data = utils.data_prepare()

    # Parameters definition. Please adjust params and choose a network
    batch_size = 64
    epochs = 25
    output_size = 10
    optimizer = 'adam'
    verbose = 1
    flag_visualizations = 1
    '''You cab choose (copy to the next line) one of the following:
    (1)'lenet' (no BN) ; (2)'lenet_bn1' (first conv layer has BN) ; (3)'lenet_bn2' (first+second conv layers have BN);
    (4)lenet_fc_bn2 (conv+first FC layer have BN) ; (5)'lenet_fc_bn2' (all layers with BN);'''
    network = 'lenet_bn1'
    # Model creation and training
    history = models_handling.model_execution(network=network,
                                              data=data,
                                              output_size=output_size,
                                              batch_size=batch_size,
                                              optimizer=optimizer,
                                              epochs=epochs,
                                              verbose=verbose)

    # Visualizations of the training process
    if flag_visualizations:
        visualizations.plot_accuracy(history)
        visualizations.plot_loss(history)


if __name__ == '__main__':
    main()

