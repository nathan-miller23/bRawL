import torch
from torch import nn
from architectures.utils import Model


class ContQNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model = Model(model_config)

    def forward(self, states, actions):
        return self.model(torch.cat([states, actions], 1))


class DiscreteQNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model = Model(model_config)

    def forward(self, states, actions):
        return self.model(states).gather(1, actions.unsqueeze(1))


class ContTwinQNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.q_net1 = ContQNet(model_config)
        self.q_net2 = ContQNet(model_config)

    def forward(self, states, actions):
        q1_out, q2_out = self.q_net1(states, actions), self.q_net2(states, actions)
        return torch.min(q1_out, q2_out), q1_out, q2_out


class DiscreteTwinQNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.q_net1 = ContQNet(model_config)
        self.q_net2 = ContQNet(model_config)

    def forward(self, states, actions):
        q1_out, q2_out = self.q_net1(states, actions), self.q_net2(states, actions)
        return torch.min(q1_out, q2_out), q1_out, q2_out


class ValueNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.v_net = Model(model_config)

    def forward(self, states):
        return self.v_net(states)
