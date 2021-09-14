This project is an implementation of batch normalization in LeNet network using tensorflow.keras on mnist dataset. The goal is to learn and characterize batch normalization impact and 
The projects contains the following files and directories:
  a. main.py - for the execution it is needed to choose the network for training and evaluation from the following possibilities:
    (1)'lenet' - the original LeNet network with no batch normalization involved;
    (2)'lenet_bn1' - the same network but adding batch normalization on the first convolutional layer;
    (3)'lenet_bn2' - the same as (2) but adding batch normalization also on the the second convolutional layer;
    (4)lenet_fc_bn1 - the same as (3) but adding batch normalization on the first fully connected layer; 
    (5)'lenet_fc_bn2' - the same as (4) but adding batch normalization also on the second fully connected layer.
