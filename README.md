This project is an implementation of batch normalization in LeNet network using tensorflow.keras on mnist dataset. The goal is to learn and characterize batch normalization impact on a neural network.

The projects contains the following files and directories:

    a. main.py - for the execution it is needed to choose the network for training and evaluation from the following possibilities:
  
    (1)'lenet' - the original LeNet network with no batch normalization involved;
    (2)'lenet_bn1' - the same network but adding batch normalization on the first convolutional layer;
    (3)'lenet_bn2' - the same as (2) but adding batch normalization also on the the second convolutional layer;
    (4)lenet_fc_bn1 - the same as (3) but adding batch normalization on the first fully connected layer; 
    (5)'lenet_fc_bn2' - the same as (4) but adding batch normalization also on the second fully connected layer.
    
    b. models_handling.py - lower level than main. Here the selected model is recognized, compiled, trained and evaluated.
    c. lenet.py - where the network classes are defined. Here you can find the original network, and the variations that include the batch normalization. 
    d. utils.py - where the instruments for mainly loading and preprocessing the data.
    e. visualizations.py - where the functions for plotting accuracy and loss graphs are.

