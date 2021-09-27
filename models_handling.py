from lenet import LeNet, LeNetBN1, LeNetBN2, LeNetFCBN1, LeNetFCBN2, lenet_keras_BN


def model_selection(network, input_shape, output_size, window_size):
    if network == 'lenet':
        print("Chosen network is LeNet")
        lenet_model = LeNet(input_shape=input_shape, output_size=output_size)
    elif network == 'lenet_bn1':
        print("Chosen network is LeNet with Batchnorm on first convolution layer")
        lenet_model = LeNetBN1(input_shape=input_shape, output_size=output_size, window=window_size)
    elif network == 'lenet_bn2':
        print("Chosen network is LeNet with Batchnorm on first + second convolution layers")
        lenet_model = LeNetBN2(input_shape=input_shape, output_size=output_size, window=window_size)
    elif network == 'lenet_fc_bn1':
        print("Chosen network is LeNet with Batchnorm on first + second convolution layers + fully connected layer")
        lenet_model = LeNetFCBN1(input_shape=input_shape, output_size=output_size, window=window_size)
    elif network == 'lenet_fc_bn2':
        print("Chosen network is LeNet with Batchnorm on first + second convolution layers + 2 fully connected layers")
        lenet_model = LeNetFCBN2(input_shape=input_shape, output_size=output_size, window=window_size)
    else:
        print("Chosen network is LeNet with Keras built-in Batch normalization at all layers")
        lenet_model = lenet_keras_BN(input_shape=input_shape)

    return lenet_model


def model_execution(network, data, output_size, batch_size, optimizer, epochs, window_size, verbose):
    """This function executes all the steps: model creation, compilation, training, and evaluation"""
    x_train, x_val, x_test, y_val, y_train, y_test = data
    input_shape = x_train.shape[1:]
    lenet_model = model_selection(network, input_shape, output_size, window_size)
    lenet_model.model_compilation(optimizer=optimizer)
    history = lenet_model.train(x_train, y_train, x_val, y_val,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose)
    lenet_model.evaluation(x_test, y_test, verbose=verbose)

    return history

