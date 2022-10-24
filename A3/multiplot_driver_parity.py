# Basic python imports for logging and sequence generation
import itertools
import pdb
import random
import logging
from tqdm import tqdm
import tqdm

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# Set random seed for python and torch to enable reproducibility (at least on the same hardware)
# random.seed(42)
# torch.manual_seed(42)

# Determine if a GPU is available for use, define as global variable
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# Main Driver Loop
def main():
    # Build the model and put it on the GPU
    logging.info("Building model")
    logging.info("Training model")
    val_accs = []
    for h in tqdm.tqdm(range(3, 20, 3)):
        maximum_training_sequence_length = h
        train = Parity(split='train', max_length=maximum_training_sequence_length)
        train_loader = DataLoader(train, batch_size=100, shuffle=True, collate_fn=pad_collate)
        hidden_dim = 64
        num_layers = 1
        model = ParityLSTM(1, hidden_dim=hidden_dim, num_layers=num_layers)
        model.to(dev)  # move to GPU if cuda is enabled
        train_model(model, train_loader, epochs=6000, lr=0.0003)

        logging.info("Running generalization experiment")
        # runParityExperiment(model, maximum_training_sequence_length)
        ## testing on the validation set
        logging.info("Starting parity experiment with model: " + str(model))
        lengths = []
        accuracy = []
        logging.info("Evaluating over strings of length 1-20.")
        k = 1
        val_acc = 1
        while k <= 20:
            val = Parity(split='val', max_length=k)
            val_loader = DataLoader(val, batch_size=1000, shuffle=False, collate_fn=pad_collate)
            val_loss, val_acc = validation_metrics(model, val_loader)
            lengths.append(k)
            k1 = val_acc.cpu().numpy()
            accuracy.append(k1)

            logging.info("length=%d val accuracy %.3f" % (k, val_acc))
            k += 1
        plt.plot(lengths, accuracy, c=get_cmap(h), label="seq len=" + str(h))
        val_accs.append(accuracy)
    plt.legend()
    plt.xlabel("Binary String Length")
    plt.ylabel("Accuracy")
    plt.show()
    plt.savefig(str(model) + '_parity_generalization.png')


######################################################################
# Task 2.2
######################################################################

# Implement a LSTM model for the parity task.

class ParityLSTM(torch.nn.Module):

    # __init__ builds the internal components of the model (presumably an LSTM and linear layer for classification)
    # The LSTM should have hidden dimension equal to hidden_dim

    def __init__(self, input_size, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # input_size, hidden_size, num_layers
        self.rnn = nn.LSTM(input_size, hidden_dim, self.num_layers, batch_first=True, bidirectional=False) #, dropout=0.07)
        # self.lin = nn.LazyLinear(62)
        self.lin = nn.LazyLinear(hidden_dim,1)
        # self.lin2 = nn.LazyLinear(hidden_dim,1)
        # self.smax = nn.Sigmoid()

    # forward runs the model on an B x max_length x 1 tensor and outputs a B x 2 tensor representing a score for
    # even/odd parity for each element of the batch
    #
    # Inputs:
    #   x -- a batch_size x max_length x 1 binary tensor. This has been padded with zeros to the max length of
    #        any sequence in the batch.
    #   s -- a batch_size x 1 list of sequence lengths. This is useful for ensuring you get the hidden state at
    #        the end of a sequence, not at the end of the padding
    #
    # Output:
    #   out -- a batch_size x 2 tensor of scores for even/odd parity

    def forward(self, x, s):
        # num_layers, x.size(0), self.hidden_size
        # Forward propagate LSTM
        # h0 = torch.zeros((self.num_layers, 1, self.hidden_dim)).to(0)
        # c0 = torch.zeros((self.num_layers, 1, self.hidden_dim)).to(0)
        # print("input ", x.shape)
        # input reshaped for LSTM
        # batch_size, seq_len, features
        x = x.view(x.shape[0], x.shape[1], 1)
        out_lstm, (h0, c0) = self.rnn(x, None)
        si = torch.tensor(s) - 1
        rs = torch.arange(0, out_lstm.shape[0])
        ones_i_want = out_lstm[rs, si, :]
        out = self.lin(ones_i_want)
        # out = self.lin2(out)
        # out = F.softmax(out)
        return out

    def __str__(self):
        return "LSTM-" + str(self.hidden_dim)


######################################################################


# This function evaluate a model on binary strings ranging from length 1 to 20.
# A plot is saved in the local directory showing accuracy as a function of this length
def runParityExperiment(model, max_train_length):
    logging.info("Starting parity experiment with model: " + str(model))
    lengths = []
    accuracy = []
    logging.info("Evaluating over strings of length 1-20.")
    k = 1
    val_acc = 1
    while k <= 20:
        val = Parity(split='val', max_length=k)
        val_loader = DataLoader(val, batch_size=1000, shuffle=False, collate_fn=pad_collate)
        val_loss, val_acc = validation_metrics(model, val_loader)
        lengths.append(k)
        k1 = val_acc.cpu().numpy()
        accuracy.append(k1)

        logging.info("length=%d val accuracy %.3f" % (k, val_acc))
        k += 1
    # print(type(accuracy))
    plt.plot(lengths, accuracy)
    plt.axvline(x=max_train_length, c="k", linestyle="dashed")
    plt.xlabel("Binary String Length")
    plt.ylabel("Accuracy")
    plt.savefig(str(model) + '_parity_generalization.png')


# Dataset of binary strings, during training generates up to length max_length
# during validation, just create sequences of max_length
class Parity(Dataset):

    def __init__(self, split="train", max_length=4):
        if split == "train":
            self.data = []
            for i in range(1, max_length + 1):
                self.data += [torch.FloatTensor(seq) for seq in itertools.product([0, 1], repeat=i)]
        else:
            self.data = [torch.FloatTensor(seq) for seq in itertools.product([0, 1], repeat=max_length)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x = self.data[idx]
        y = x.sum() % 2
        return x, y

    # Function to enable batch loader to concatenate binary strings of different lengths and pad them


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]

    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    # pdb.set_trace()
    # yy = torch.tensor(yy, dtype=torch.int, device=0)
    yy = torch.tensor(yy)
    return xx_pad, yy, x_lens


# Basic training loop for cross entropy loss
def train_model(model, train_loader, epochs=2000, lr=0.0003):
    # Define a cross entropy loss function
    crit = torch.nn.CrossEntropyLoss()

    # Collect all the learnable parameters in our model and pass them to an optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Adam is a version of SGD with dynamic learning rates
    # (tends to speed convergence but often worse than a well tuned SGD schedule)
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)

    # Main training loop over the number of epochs
    for i in range(epochs):

        # Set model to train mode so things like dropout behave correctly
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0

        # for each batch in the dataset
        for j, (x, y, l) in enumerate(train_loader):
            # push them to the GPU if we are using one
            x = x.to(dev)
            y = y.type(torch.LongTensor)
            y = y.to(dev)
            # predict the parity from our model
            # if(x.size(0)>30):
            y_pred = model(x, l)

            # compute the loss with respect to the true labels
            loss = crit(y_pred, y)

            # zero out the gradients, perform the backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # compute loss and accuracy to report epoch level statitics
            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        if i % 10 == 0:
            logging.info("epoch %d train loss %.3f, train acc %.3f" % (
            i, sum_loss / total, correct / total))  # , val_loss, val_acc))


def validation_metrics(model, loader):
    # set the model to evaluation mode to turn off things like dropout
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    crit = torch.nn.CrossEntropyLoss()
    # CrossEntropyLoss
    for i, (x, y, l) in enumerate(loader):
        x = x.to(dev)
        y = y.type(torch.LongTensor)
        y = y.to(dev)
        y_hat = model(x, l)
        # print("y_shape ", y.shape)
        loss = crit(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]

    return sum_loss / total, correct / total


main()