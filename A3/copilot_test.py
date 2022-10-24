import torch.nn as nn
import torch.nn.functional as F
import torch

class Predict(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Predict, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x):
        return self.forward(x)

dataset =  torch.load('dataset.pt')
dataloader = torch.utils.data.DataLoader(dataset=None, batch_size=1, shuffle=False)

for batch in dataloader:
    print(batch)
