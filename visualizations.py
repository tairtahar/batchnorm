import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(history):
    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.yticks(np.arange(0.5,1,0.02))
    plt.grid(which='major', linestyle=':', alpha=0.6)
    plt.ylim([0.7, 1])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig('accuracy.pdf')
    plt.show()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig('loss.pdf')
    plt.show()


def plot_accuracies(histories, networks):
    for i in range(len(histories)):
        plt.plot(histories[i].history['accuracy'])
    plt.yticks(np.arange(0.5, 1, 0.02))
    plt.grid(which='major', linestyle=':', alpha=0.6)
    plt.ylim([0.7, 1])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(networks, loc='lower right')
    # plt.savefig('accuracy.pdf')
    plt.show()