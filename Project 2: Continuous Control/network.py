import numpy as np

import torch
import torch.nn as nn

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        limit1  = 1./np.sqrt(state_size*fc1_units)
        self.fc1.weight.data.uniform_(-limit1,limit1)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        limit2  = 1./np.sqrt(fc1_units*fc2_units)
        self.fc2.weight.data.uniform_(-limit2,limit2)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear((state_size+action_size), fc1_units)
        limit1  = 1./np.sqrt((state_size+action_size)*fc1_units)
        self.fc1.weight.data.uniform_(-limit1,limit1)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        limit2  = 1./np.sqrt(fc1_units*fc2_units)
        self.fc2.weight.data.uniform_(-limit2,limit2)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)