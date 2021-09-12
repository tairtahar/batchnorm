from lenet2 import LeNet, LeNetBN1, LeNetBN2, LeNetFCBN1


def model_selection(network, input_shape, output_size, batch_size):
    if network == 'lenet':
        lenet_model = LeNet(input_shape=input_shape, output_size=output_size)
    elif network == 'lenet_bn1':
        lenet_model = LeNetBN1(input_shape=input_shape, batch_size=batch_size, output_size=output_size)
    elif network == 'lenet_bn2':
        lenet_model = LeNetBN2(input_shape=input_shape, batch_size=batch_size, output_size=output_size)
    elif network == 'lenet_fc_bn1':
        lenet_model = LeNetFCBN1(input_shape=input_shape, batch_size=batch_size, output_size=output_size)

    return lenet_model