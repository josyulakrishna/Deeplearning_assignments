import pdb
import random
from turtle import width
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy.matlib as mat
from sklearn.utils import shuffle
# from sklearn.metrics import accuracy_score

font = {'weight' : 'normal','size'   : 22}
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
    # return (exps / np.sum(exps) + 0.000001)
    return exps/np.sum(np.exp(X - np.max(X)), axis=1, keepdims=True)

  def forward(self, logits, labels):
    self.y = np.argmax(labels, axis=1)  # labels
    la = logits[np.arange(len(logits)), self.y]
    # loss = -(la-np.max(la)) + np.log(np.sum(np.exp(logits - np.max(logits)), axis=-1))
    loss = -(la-np.max(logits)) + np.log(np.sum(np.exp(logits-np.max(logits)), axis=1, keepdims=True))


    return loss


  # Compute the gradient of the cross entropy loss with respect to the the input logits
  def backward(self,logits,labels):
    labels = np.argmax(labels, axis=1)
    oa = np.zeros_like(logits)
    oa[np.arange(len(logits)), labels] = 1
    # softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    softmax = self.stable_softmax(logits)
    return (- oa + softmax) / logits.shape[0]

class ReLU:

  # Compute ReLU(input) element-wise
  def forward(self, input):
      self.X = input
      return np.maximum(input,0)

  # Given dL/doutput, return dL/dinput
  def backward(self, act, grad):
      relu_grad = act>0
      return grad*relu_grad

  # No parameters so nothing to do during a gradient descent step
  def step(self,step_size):
      return


class LinearLayer:
  # Initialize our layer with (input_dim, output_dim) weight matrix and a (1,output_dim) bias vector
  def __init__(self, input_dim, output_dim):
    self.grad_biases = None
    self.grad_weights = None
    self.W = np.random.randn(input_dim, output_dim)*np.sqrt(1. / output_dim)
    self.b = np.random.randn(output_dim)
    self.m_dw, self.v_dw = 0.1, 0.1
    self.m_db, self.v_db = 0.1, 0.1
    self.beta1 = 0.9
    self.epsilon = 1e-8
    self.eta = 0.0001
    
  # During the forward pass, we simply compute XW+b
  def forward(self, input):
    self.input = input
    return np.dot(input, self.W)+self.b

  def backward(self, act, grad):
    # print("backward linear layer")
    # pdb.set_trace()
    # dL/dW=dL/dZ*dZ/dW
    grad_input = np.dot(grad, np.transpose(self.W))
    self.grad_weights = np.transpose(np.dot(np.transpose(grad), act))
    self.grad_bias = np.sum(grad, axis=0)
    return grad_input

  ######################################################
  # Q5 Implement ADAM with Weight Decay
  ######################################################  
  def step(self, eta):
    t = 1
    self.eta = eta
    self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * self.grad_weights
    # *** biases *** #
    self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * self.grad_bias

    ## rms beta 2
    # *** weights *** #
    self.v_dw = self.beta1 * self.v_dw + (1 - self.beta1) * (self.grad_weights**2)
    # *** biases *** #
    self.v_db = self.beta1 * self.v_db + (1 - self.beta1) * (self.grad_bias**2)

    ## bias correction
    m_dw_corr = self.m_dw / (1 - self.beta1 ** t)
    m_db_corr = self.m_db / (1 - self.beta1 ** t)
    v_dw_corr = self.v_dw / (1 - self.beta1 ** t)
    v_db_corr = self.v_db / (1 - self.beta1 ** t)

    ## update weights and biases
    self.W = self.W - self.eta * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
    self.b = self.b - self.eta * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  one_hot_Y[np.arange(Y.size), Y.ravel()] = 1
  # one_hot_Y = one_hot_Y.T
  return one_hot_Y


######################################################
# Q6 Implement Evaluation and Training Loop
######################################################
def accuracy_score(true_values, predictions):
  N = true_values.shape[0]
  accuracy = (true_values == predictions).sum() / N
  return  accuracy


# Given a model, X/Y dataset, and batch size, return the average cross-entropy loss and accuracy over the set
def evaluate(model, X_val, Y_val, batch_size):
  activations = []
  loss_func = CrossEntropySoftmax()
  batch_loss = []
  accuracy = []
  for x_batch, y_batch in iterate_minibatches(X_val, Y_val, batchsize=batch_size, shuffle=True):
    activations = model.forward(x_batch)
    loss = loss_func.forward(activations[-1], y_batch)
    pred_ = activations[-1].argmax(axis=-1)
    actual_labels = np.argmax(y_batch, axis=1)
    batch_loss.append(loss)
    accuracy.append(accuracy_score(actual_labels, pred_))
  batch_loss = np.array(batch_loss)
  accuracy = np.array(accuracy)
  avg_loss = np.mean(batch_loss)
  avg_acc = np.mean(accuracy)
  return avg_loss, avg_acc

#function to iterated over minibadges.
def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main():

  # Set optimization parameters (NEED TO CHANGE THESE)
  batch_size = 60
  max_epochs = 20
  step_size = 0.001

  number_of_layers = 5
  width_of_layers = 512
  # 0.791

  # Load data
  X_train, Y_train, X_val, Y_val, X_test, Y_test = loadCIFAR10Data()
  X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
  X_val = (X_val - np.min(X_val)) / (np.max(X_val) - np.min(X_val))
  X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
  # pdb.set_trace()
  
  # Some helpful dimensions
  num_examples, input_dim = X_train.shape
  output_dim = 3 # number of class labels


  # Build a network with input feature dimensions, output feature dimension,
  # hidden dimension, and number of layers as specified below. You can edit this as you please.
  net = FeedForwardNeuralNetwork(input_dim, output_dim, width_of_layers, number_of_layers)
  #CE LOSS
  loss_func = CrossEntropySoftmax()
  # Some lists for book-keeping for plotting later
  losses = []
  val_losses = []
  accs = []
  val_accs = []
  

  for e in range(max_epochs):
      # Scramble order of examples
      # Gather batch
      epoch_loss = []
      for x_batch, y_batch in iterate_minibatches(X_train, Y_train, batchsize=batch_size, shuffle=True):
        # Compute forward pass
        activations = net.forward(x_batch)
        #activations has all values upto last linear layer
        # Compute loss for CrossEntropySoftmax
        loss_func.forward(activations[-1], y_batch)
        # Backward loss and networks
        grad = loss_func.backward(activations[-1], y_batch)
        net.backward(activations, grad)
        # Take optimizer step
        net.step(step_size)
      loss_, acc = evaluate(net, x_batch, y_batch, batch_size)
      losses.append(loss_)
      accs.append(acc)
      # Book-keeping for loss / accuracy
      print("Iter {0}, batch size {1}, loss is {2} ".format(e, batch_size, np.sum(loss_)/3.0))
      # Evaluate performance on validation.
      vl, va = evaluate(net, X_val, Y_val, batch_size)
      val_losses.append(vl)
      val_accs.append(va)
      print("Iter {0}, val loss is {1}, val accuracy is {2} ".format(e,  vl, va))
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
  fig, ax1 = plt.subplots(figsize=(16,9))
  color = 'tab:red'
  ax1.plot(range(len(losses)), losses, c=color, alpha=0.25, label="Train Loss")
  ax1.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_losses))], val_losses,c="red", label="Val. Loss")
  ax1.set_xlabel("Iterations")
  ax1.set_ylabel("Avg. Cross-Entropy Loss", c=color)
  ax1.tick_params(axis='y', labelcolor=color)
  #ax1.set_ylim(-0.01,3)
  
  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:blue'
  ax2.plot(range(len(losses)), accs, c=color, label="Train Acc.", alpha=0.25)
  ax2.plot([np.ceil((i+1)*len(X_train)/batch_size) for i in range(len(val_accs))], val_accs,c="blue", label="Val. Acc.")
  ax2.set_ylabel(" Accuracy", c=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.set_ylim(-0.01,1.01)
  
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  ax1.legend(loc="center")
  ax2.legend(loc="center right")
  plt.show()  


  ################################
  # Q7 Tune and Evaluate on Test
  ################################
  _, tacc = evaluate(net, X_test, Y_test, len(Y_test))
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
      for i in range(num_layers-2):
        self.layers.append(LinearLayer(hidden_dim, hidden_dim))
        self.layers.append(ReLU())
      self.layers.append(LinearLayer(hidden_dim, output_dim))

  def forward(self, X):
    activations = []
    for layer in self.layers:
      X = layer.forward(X)
      activations.append(X)
    return activations

  def backward(self, activations, grad):
    #####NOTE TO SELF:::::############
    ####DOES NOT CONTAIN CROSSENTROPYLAYER GRADIENT DO NOT FORGET
    #####PASS THE GRADIENT TO THIS FUNCTION AND DONT WASTE TIME######
    # this starts from linear layer ##
    # therefore get CEGRAD
    # for layer in reversed(self.layers):
    #   grad = layer.backward(grad)
    for i in range(1, len(self.layers)):
      # print(len(self.layers) - i, len(self.layers) - i-1)
      grad = self.layers[len(self.layers) - i].backward(activations[len(self.layers) - i - 1], grad)
      # print(len(grad))

  def step(self, step_size):
    # for layer in self.layers:
    #   layer.step(step_size)
    for i in range(1, len(self.layers)):
      # print("layer ", len(self.layers) - i)
      self.layers[len(self.layers) - i].step(step_size)

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
  r = x[:1024].reshape(32,32)
  g = x[1024:2048].reshape(32,32)
  b = x[2048:].reshape(32,32)
  
  plt.imshow(np.stack([r,g,b],axis=2))
  plt.axis('off')
  plt.show()


if __name__=="__main__":
  main()