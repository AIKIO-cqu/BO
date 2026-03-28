import torch
import torch.nn as nn


def init_linear(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(torch.tanh(x))


class DeterministicActor(nn.Module):
    def __init__(self, s_dim,  a_dim, hidden):
        super(DeterministicActor, self).__init__()
        self.actor = nn.Sequential(nn.Linear(s_dim, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, hidden, bias=False),
                                   nn.Tanh(),
                                   nn.Linear(hidden, a_dim, bias=False),
                                   nn.Tanh())
        init_linear(self)

    def forward(self, s):
        return self.actor(s)


class DoubleCritic(nn.Module):
    def __init__(self, s_dim, a_dim, hidden):
        super(DoubleCritic, self).__init__()
        self.q1 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, 1, bias=False))
        self.q2 = nn.Sequential(nn.Linear(s_dim + a_dim, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, hidden, bias=False),
                                Abs(),
                                nn.Linear(hidden, 1, bias=False))
        init_linear(self)

    def forward(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q1 = self.q1(s_a)
        q2 = self.q2(s_a)
        return q1, q2

    def Q1(self, s, a):
        s_a = torch.cat([s, a], dim=-1)
        q1 = self.q1(s_a)
        return q1
