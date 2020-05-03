import torch.nn as nn
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear()
        self.lin2 = nn.Linear()
    def forward(self, document):
        out =

