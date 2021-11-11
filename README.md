![TF](https://img.shields.io/badge/TensorFlow-2.6.0-yellowgreen)  ![python](https://img.shields.io/badge/Python-3.6-orange)


# Batch Normalization Over LeNet
This project is an implementation from scratch of batch normalization in *LeNet* network using *tensorflow.keras* on *mnist* dataset. The goal is to learn and characterize batch normalization impact on a neural network performance.

![LeNet](/LeNet.PNG)



### Main arguments are possible to adjust:

| Parameter           | Explanation                              | Default    |
| ------------------- | ---------------------------------------- | ---------- |
| batch_size          | Batch size for training                  | 64         |
| output_size         | output size, number of classes           | 10         |
| epochs              | number of epochs for training            | 10         |
| optimizer           | optimizer for training                   | 'adam'     |
| epsilon             | number of epochs for moving standardization during inference time | 0.00000001 |
| window_size         | window size for averaging in batchnorm algorithm | 5          |
| verbose             | verbosity                                | 1          |
| flag_visualizations | flag for presenting plots of the training process | 1          |

## Project Structure

1. main.py - for the execution it is needed to choose the network for training and evaluation from the following possibilities:
   1. 'lenet' - the original LeNet network with no batch normalization involved;
   2. 'lenet_bn1' - the same network but adding batch normalization on the first convolutional layer;
   3. 'lenet_bn2' - the same as (2) but adding batch normalization also on the the second convolutional layer;
   4. lenet_fc_bn1 - the same as (3) but adding batch normalization on the first fully connected layer; 
   5. 'lenet_fc_bn2' - the same as (4) but adding batch normalization also on the second fully connected layer.


2. models_handling.py - lower level than main. Here the selected model is recognized, compiled, trained and evaluated.
3. lenet.py - where the network classes are defined. Here you can find the original network, and the variations that include the batch normalization. 
4. utils.py - functions mainly for loading and preprocessing the data.
   e. visualizations.py - where the functions for plotting accuracy and loss graphs are.
5. results - where it is possible to visualize the training and validation accuracy during the training of each of the networks used.



### Citing

> Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." In *International conference on machine learning*, pp. 448-456. PMLR, 2015.

 ```latex
 @inproceedings{ioffe2015batch,
   title={Batch normalization: Accelerating deep network training by reducing internal covariate shift},
   author={Ioffe, Sergey and Szegedy, Christian},
   booktitle={International conference on machine learning},
   pages={448--456},
   year={2015},
   organization={PMLR}
 }
 ```



> LeCun, Yann, Léon Bottou, Yoshua Bengio, and Patrick Haffner. "Gradient-based learning applied to document recognition." *Proceedings of the IEEE* 86, no. 11 (1998): 2278-2324.

```latex
@ARTICLE{726791,
  author={Lecun, Y. and Bottou, L. and Bengio, Y. and Haffner, P.},
  journal={Proceedings of the IEEE}, 
  title={Gradient-based learning applied to document recognition}, 
  year={1998},
  volume={86},
  number={11},
  pages={2278-2324},
  doi={10.1109/5.726791}}

```



