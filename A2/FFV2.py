import pdb
import random
from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy.matlib as mat
from sklearn.utils import shuffle

font = {'weight': 'normal', 'size': 22}
matplotlib.rc('font', **font)
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


######################################################
# Q4 Implement Init, Forward, and Backward For Layers
######################################################


class CrossEntropySoftmax:
    # Compute the cross entropy loss after performing softmax
    # logits -- batch_size x num_classes set of scores, logits[i,j] is score of class j for batch element i
    # labels -- batch_size x 1 vector of integer label id (0,1,2) where labels[i] is the label for batch element i
    #
    # Output should be a positive scalar value equal to the average cross entropy loss after softmax
    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X))
        return (exps / np.sum(exps) + 0.000001)

    def forward(self, logits, labels):
        self.X = logits
        self.y = np.argmax(labels, axis=1)  # labels
        m = labels.shape[0]
        p = self.stable_softmax(logits)
        print("forward")
        pdb.set_trace()
        log_likelihood = -np.log(p[range(m), self.y])
        loss = np.sum(log_likelihood) / m
        return loss

    # Compute the gradient of the cross entropy loss with respect to the the input logits
    def backward(self):
        m = self.y.shape[0]
        grad = self.stable_softmax(self.X)
        print("backward")
        pdb.set_trace()
        grad[range(m), self.y] -= 1
        grad = grad / m
        return grad


class ReLU:

    # Compute ReLU(input) element-wise
    def forward(self, input):
        return np.maximum(input, 0)

    # Given dL/doutput, return dL/dinput
    def backward(self, grad):
        grad[grad <= 0] = 0
        grad[grad > 0] = 1
        return grad

    # No parameters so nothing to do during a gradient descent step
    def step(self, step_size):
        return


class LinearLayer:

    # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
    def __init__(self, input_dim, output_dim):
        # raise Exception('Student error: You haven\'t implemented the init for LinearLayer yet.')
        # self.input_dim = input_dim
        # self.output_dim = output_dim
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(1, output_dim)
        # self.gradWeight = np.zeros_like(self.W)
        # self.gradBias = np.zeros_like(self.b)
        self.input = np.zeros_like(input_dim)
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = 0.9
        self.beta2 = 0.9
        self.epsilon = 1e-8
        self.eta = 0.01

    # During the forward pass, we simply compute XW+b
    def forward(self, input):
        self.input = input
        return np.dot(input, self.W) + self.b

    # Inputs:
    #
    # grad dL/dZ -- For a batch size of n, grad is a (n x output_dim) matrix where
    #         the i'th row is the gradient of the loss of example i with respect
    #         to z_i (the output of this layer for example i)

    # Computes and stores:
    #
    # self.grad_weights dL/dW --  A (input_dim x output_dim) matrix storing the gradient
    #                       of the loss with respect to the weights of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the weights.
    #
    # self.grad_bias dL/db-- A (1 x output_dim) matrix storing the gradient
    #                       of the loss with respect to the bias of this layer.
    #                       This is an summation over the gradient of the loss of
    #                       each example with respect to the bias.

    # Return Value:
    #
    # grad_input dL/dX -- For a batch size of n, grad_input is a (n x input_dim) matrix where
    #               the i'th row is the gradient of the loss of example i with respect
    #               to x_i (the input of this layer for example i)

    def backward(self, grad):
        print("backward linear layer")
        pdb.set_trace()

        self.grad_weights = np.dot(self.input.T, grad)
        self.grad_bias = self.input
        # grad_input = np.concatenate((self.grad_weights,self.grad_bias))
        return self.grad_weights

    ######################################################
    # Q5 Implement ADAM with Weight Decay
    ######################################################
    def step(self, step_size):
        t = 1
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * self.grad_weights
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * self.grad_bias

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (self.grad_weights ** 2)
        # *** biases *** #
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (self.grad_biasd)

        ## bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
        m_db_corr = self.m_db / (1 - self.beta1 ** t)
        v_dw_corr = self.v_dw / (1 - self.beta2 ** t)
        v_db_corr = self.v_db / (1 - self.beta2 ** t)

        ## update weights and biases
        self.w = self.W - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        self.b = self.b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y.ravel()] = 1
    # one_hot_Y = one_hot_Y.T
    return one_hot_Y


######################################################
# Q6 Implement Evaluation and Training Loop
######################################################

# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
    raise Exception('Student error: You haven\'t implemented the step for evalute function.')


def main():
    # Set optimization parameters (NEED TO CHANGE THESE)
    batch_size = 1
    max_epochs = 1
    step_size = 1

    number_of_layers = 2
    width_of_layers = 100

    # Load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = loadCIFAR10Data()
    pdb.set_trace()

    # Some helpful dimensions
    num_examples, input_dim = X_train.shape
    output_dim = 3  # number of class labels

    # Build a network with input feature dimensions, output feature dimension,
    # hidden dimension, and number of layers as specified below. You can edit this as you please.
    net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)
    # CE LOSS
    loss_func = CrossEntropySoftmax()
    # Some lists for book-keeping for plotting later
    losses = []
    val_losses = []
    accs = []
    val_accs = []

    for e in range(max_epochs):
        # Scramble order of examples

        X_train, y_train = shuffle(X_train, Y_train, random_state=random.randint(0, 9))
        X_batch = X_train[:batch_size, :]
        Y_batch = Y_train[:batch_size, :]
        # for each batch in data:

        # Gather batch

        # Compute forward pass
        y_out = net.forward(X_batch)

        # Compute loss
        loss_func.forward(y_out, Y_batch)

        # Backward loss and networks
        # grad=loss_func.backward()
        net.backward(1)
        # Take optimizer step
        net.step(0.01)
        # Book-keeping for loss / accuracy

    # Evaluate performance on validation.

    ###############################################################
    # Print some stats about the optimization process after each epoch
    ###############################################################
    # epoch_avg_loss -- average training loss across batches this epoch
    # epoch_avg_acc -- average accuracy across batches this epoch
    # vacc -- validation accuracy this epoch
    ###############################################################

    # logging.info("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%      Val Acc:  {:8.4}%".format(i,epoch_avg_loss, epoch_avg_acc, vacc*100))

    ###############################################################
    # Code for producing output plot requires
    ###############################################################
    # losses -- a list of average loss per batch in training
    # accs -- a list of accuracies per batch in training
    # val_losses -- a list of average validation loss at each epoch
    # val_acc -- a list of validation accuracy at each epoch
    # batch_size -- the batch size
    ################################################################

    # Plot training and validation curves
    fig, ax1 = plt.subplots(figsize=(16, 9))
    color = 'tab:red'
    ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
    ax1.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_losses))], val_losses, c="red",
             label="Val. Loss")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.set_ylim(-0.01,3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
    ax2.plot([np.ceil((i + 1) * len(X_train) / batch_size) for i in range(len(val_accs))], val_accs, c="blue",
             label="Val. Acc.")
    ax2.set_ylabel(" Accuracy", c=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.01, 1.01)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="center")
    ax2.legend(loc="center right")
    plt.show()

    ################################
    # Q7 Tune and Evaluate on Test
    ################################
    _, tacc = evaluate(net, X_test, Y_test, batch_size)
    print(tacc)


#####################################################
# Feedforward Neural Network Structure
# -- Feel free to edit when tuning
#####################################################

class FeedForwardNeuralNetwork:

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):

        if num_layers == 1:
            self.layers = [LinearLayer(input_dim, output_dim)]
        else:
            self.layers = [LinearLayer(input_dim, hidden_dim)]
            self.layers.append(ReLU())
            for i in range(num_layers - 2):
                self.layers.append(LinearLayer(hidden_dim, hidden_dim))
                self.layers.append(ReLU())
            self.layers.append(LinearLayer(hidden_dim, output_dim))

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, step_size):
        for layer in self.layers:
            layer.step(step_size)


#####################################################
# Utility Functions for Loading and Visualizing Data
#####################################################

def loadCIFAR10Data():
    with open("cifar10_hst_train", 'rb') as fo:
        data = pickle.load(fo)
    X_train = data['images']
    Y_train = data['labels']
    Y_train = one_hot(Y_train)

    with open("cifar10_hst_val", 'rb') as fo:
        data = pickle.load(fo)
    X_val = data['images']
    Y_val = data['labels']
    Y_val = one_hot(Y_val)
    # Y_val = Y_val.T

    with open("cifar10_hst_test", 'rb') as fo:
        data = pickle.load(fo)
    X_test = data['images']
    Y_test = data['labels']
    Y_test = one_hot(Y_test)
    # Y_test = Y_test.T

    logging.info("Loaded train: " + str(X_train.shape))
    logging.info("Loaded val: " + str(X_val.shape))
    logging.info("Loaded test: " + str(X_test.shape))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def displayExample(x):
    r = x[:1024].reshape(32, 32)
    g = x[1024:2048].reshape(32, 32)
    b = x[2048:].reshape(32, 32)

    plt.imshow(np.stack([r, g, b], axis=2))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()