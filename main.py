import models_handling
import utils
import visualizations


def main():
    # Data loading and preprocessing
    data = utils.data_prepare()

    # Parameters definition. Please adjust and choose a network
    batch_size = 64
    epochs = 10
    output_size = 10
    optimizer = 'adam'
    verbose = 1
    flag_visualizations = 1
    network = 'lenet'   #  'lenet_bn1'# , 'lenet_bn1', 'lenet_fc_bn1' 'lenet' lenet_bn2

    # Model creation and training
    history = models_handling.model_execution(network, data, output_size, batch_size, optimizer, epochs, verbose)

    # Visualizations of the training process
    if flag_visualizations:
        visualizations.plot_accuracy(history)
        visualizations.plot_loss(history)


if __name__ == '__main__':
    main()

