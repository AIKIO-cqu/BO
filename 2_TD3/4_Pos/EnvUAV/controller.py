import torch
import torch.nn as nn


class Controller:
    def __init__(self, path, prefix, s_dim=2, a_dim=1, hidden=32):
        self.actor = DeterministicActor(s_dim, a_dim, hidden)
        self.actor.load_state_dict(torch.load(path + '/'+prefix+'Actor.pth'))

    def get_action(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            a = self.actor(s)
        return a.item()


class DeterministicActor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(DeterministicActor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, a_dim, bias=False),
                                   nn.Tanh())

    def forward(self, s):
        return self.actor(s)