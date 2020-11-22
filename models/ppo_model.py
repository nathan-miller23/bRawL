import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class RllibPPOModel(TorchModelV2, nn.Module):
    """
    Model that will map environment states to action probabilities. Will be shared across agents
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):

        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # params we got to pass in from the call to "run"
        custom_params = model_config["custom_model_config"]

        ## Parse custom network params
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        hidden_output_size = custom_params["HIDDEN_OUTPUT_SIZE"]

        modules = []
       # if num_convs > 0:
       #     modules.append(torch.nn.Conv2d(obs_space.shape[-1], num_filters, kernel_size=[5, 5], padding=3))
       #     modules.append(torch.nn.LeakyReLU())
       #     modules.append(torch.nn.MaxPool2d(2, stride=2, padding=1))
       # for i in range(0, num_convs-1):
       #     modules.append(torch.nn.Conv2d(num_filters, num_filters, kernel_size=[3, 3], padding=2))
       #     modules.append(torch.nn.LeakyReLU())
       #     modules.append(torch.nn.MaxPool2d(2, stride=2, padding=1))
        modules.append(nn.Flatten())
        
        #modules.append(nn.InstanceNorm1d(39))
        
        modules.append(nn.Linear(39, size_hidden_layers))
        modules.append(torch.nn.LeakyReLU())
        
        for i in range(num_hidden_layers - 1):
            modules.append(torch.nn.Linear(size_hidden_layers, size_hidden_layers))
            modules.append(torch.nn.LeakyReLU())
        self.layers = nn.Sequential(*modules)
        self._num_outputs = num_outputs
        self.model_out = torch.nn.Linear(size_hidden_layers, self._num_outputs)
        self.value_out = torch.nn.Linear(size_hidden_layers, 1)

    def forward(self, input_dict, state=None, seq_lens=None):
        obs = input_dict["obs"].float()#.permute(0,3,1,2)
        obs = obs / np.linalg.norm(obs)
        hidden = self.layers(obs)
        model_out = self.model_out(hidden).view(-1, self._num_outputs)
        self._value_out = self.value_out(hidden)
        return model_out, state

    def value_function(self):
        return torch.reshape(self._value_out, [-1])
