from lenet import LeNet, LeNetBN1, LeNetBN2, LeNetFCBN1


def model_selection(network, x_train, input_shape, output_size, batch_size):
    if network == 'lenet':
        lenet_model = LeNet(input_shape=input_shape, output_size=output_size)
    elif network == 'lenet_bn1':
        lenet_model = LeNetBN1(input_shape=input_shape, output_size=output_size)
    elif network == 'lenet_bn2':
        lenet_model = LeNetBN2(input_shape=input_shape, batch_size=batch_size, output_size=output_size)
    elif network == 'lenet_fc_bn1':
        lenet_model = LeNetFCBN1(input_shape=input_shape, batch_size=batch_size, output_size=output_size)

    return lenet_model


def model_execution(network, data, output_size, batch_size, optimizer, epochs, verbose):
    x_train, x_val, x_test, y_val, y_train, y_test = data
    input_shape = x_train.shape[1:]
    lenet_model = model_selection(network, x_train, input_shape, output_size, batch_size)
    # lenet_model.build(input_shape, output_size)
    lenet_model.model_compilation(optimizer=optimizer)
    history = lenet_model.train(x_train, y_train, x_val, y_val,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose)
    lenet_model.evaluation(x_test, y_test, verbose=verbose)

    return history

