import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel

torch, nn = try_import_torch()

class RllibDQNModel(DQNTorchModel, nn.Module):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):

        super().__init__(obs_space, action_space, num_outputs, model_config, name, **kwargs)

        self._num_outputs = num_outputs


        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_model_config"]

        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]

        modules = []
        if num_convs > 0:
            modules.append(nn.Conv1d(obs_space.shape[0], num_filters, kernel_size=3, padding=1))
            modules.append(torch.nn.LeakyReLU())
        for _ in range(num_convs-1):
            modules.append(torch.nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1))
            modules.append(torch.nn.LeakyReLU())
        modules.append(nn.Flatten())
        
        #modules.append(nn.InstanceNorm1d(39))
        in_size = num_filters * obs_space.shape[1] if num_convs > 0 else obs_space.shape[0] * obs_space.shape[1]
        
        modules.append(nn.Linear(in_size, size_hidden_layers))
        modules.append(torch.nn.LeakyReLU())
        
        for i in range(num_hidden_layers - 1):
            modules.append(torch.nn.Linear(size_hidden_layers, size_hidden_layers))
            modules.append(torch.nn.LeakyReLU())

        modules.append(nn.Linear(size_hidden_layers, self._num_outputs))

        
        self.net = nn.Sequential(*modules)
    

    def forward(self, input_dict, state=None, seq_lens=None):
        obs = input_dict["obs"].float()
        # obs = obs / np.linalg.norm(obs)
        hidden = self.net(obs)
        return hidden, state

