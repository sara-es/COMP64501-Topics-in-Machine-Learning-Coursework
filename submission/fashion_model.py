import torch
import torch.nn as nn  # neural network modules
import torch.nn.functional as F  # useful stateless functions


class Net(nn.Module):
    """
    Define your model here. Feel free to modify all code below, but do not change the class name, and make sure Net() works with no arguments.
    This simple example is a feedforward neural network with one hidden layer.
    Please note that this example model has 101,770 parameters, which is over the 100,000 limit.
    """
    def __init__(self, input_size=28*28, hidden_size=128, num_classes=10):
        super(Net, self).__init__()

        # We define the layers of our model here by instantiating layer objects.
        # Here we define two fully connected (linear) layers.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # nn.init module contains initialization methods:
        # https://docs.pytorch.org/docs/main/nn.init.html
        # This particular one is called Kaiming initialization (also known as
        # He initialization) described in He, K. et al. (2015)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        # The forward pass defines the connectivity of the layers defined in __init__.
        x = x.view(x.size(0), -1)  # flatten the input tensor (first dimension is batch size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
