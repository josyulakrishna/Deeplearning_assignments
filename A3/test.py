# Imports for Pytorch for the things we need
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms, datasets

# Imports for plotting our result curves
import matplotlib
import matplotlib.pyplot as plt
# Basic python imports for logging and sequence generation
import itertools
import random
import logging
import pickle
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np



device = torch.device('cuda')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels, out_channels, kernel_size, stride = 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(3)

    def forward(self, x):
        x = self.conv1(x)
        # Batch, output_channel, kernel
        # print("conv1 ", x.size())
        x = F.relu(x)
        # print("relu1 ", x.size())
        x = self.conv2(x)
        # print("conv2 ", x.size())
        x = F.max_pool2d(x, 2)
        # print("pool ", x.size())
        x = F.relu(x)
        # print("relu ", x.size())
        x = self.conv3(x)
        # print("conv3 ", x.size())
        x = F.relu(x)
        # print("relu ", x.size())
        x = self.conv4(x)
        # print("conv4 ", x.size())
        x = F.relu(x)
        # print("relu ", x.size())
        x = F.max_pool2d(x, 2)
        # print("pool ", x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        # _, n_chans, _, _ = x.size()
        mu = torch.zeros(512, device=0)
        std = torch.zeros(512, device=0)
        x = F.batch_norm(x, mu, std)
        # print("fc1 ", x.size())
        x = F.relu(x)
        # print("relu ", x.size())
        x = self.fc2(x)
        # print("fc2 ", x.size())
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CIFAR3(Dataset):

    def __init__(self, split="train", transform=None):
        if split == "train":
            with open("cifar10_hst_train", 'rb') as fo:
                self.data = pickle.load(fo)
        elif split == "val":
            with open("cifar10_hst_val", 'rb') as fo:
                self.data = pickle.load(fo)
        else:
            with open("cifar10_hst_test", 'rb') as fo:
                self.data = pickle.load(fo)

        self.transform = transform

    def __len__(self):
        return len(self.data['labels'])

    def __getitem__(self, idx):

        x = self.data['images'][idx, :]
        r = x[:1024].reshape(32, 32)
        g = x[1024:2048].reshape(32, 32)
        b = x[2048:].reshape(32, 32)

        x = Tensor(np.stack([r, g, b]))

        if self.transform is not None:
            x = self.transform(x)

        y = self.data['labels'][idx, 0]
        return x, y

    #########################################################


# Training and Evaluation
#########################################################

test_transform = transforms.Compose([
    transforms.Normalize(mean=[127.5, 127.5, 127.5],
                         std=[127.5, 127.5, 127.5])
])

test_data = CIFAR3("test", transform=test_transform)

batch_size = 256
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# Build model
model = Net()
PATH = "/home/josyula/Documents/Assignments/Deep Learning/A3/best_model_on_val"
# model = torch.load(PATH)
model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)
acc_log = []

for j, input in enumerate(testloader, 0):
    # print("in Test")
    x = input[0].to(device)
    y = input[1].type(torch.LongTensor).to(device)
    out = model(x)

    _, predicted = torch.max(out.data, 1)
    correct = (predicted == y).sum()
    acc_log.append(correct.item() / len(y))
    print("accuracy ", correct.item() / len(y))






