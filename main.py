import models
import utils
import visualizations


def main():
    # Data loading and preprocessing
    data = utils.data_prepare()

    # Parameters definition. Please adjust and choose a network
    batch_size = 256
    epochs = 15
    output_size = 10
    optimizer = 'sgd'
    verbose = 1
    flag_visualizations = 1
    network = 'lenet_bn1'  # , 'lenet_bn2', 'lenet_fc_bn1' 'lenet'

    # Model creation and training
    history = models.model_exexution(network, data, output_size, batch_size, optimizer, epochs, verbose)

    # Visualizations of the training process
    if flag_visualizations:
        visualizations.plot_accuracy(history)
        visualizations.plot_loss(history)


if __name__ == '__main__':
    main()

