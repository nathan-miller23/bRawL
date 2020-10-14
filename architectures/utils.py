import torch
from torch import nn


def polyak_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * param.data + target_param.data * (1.0 - tau))


def gen_noise(scale, tensor, device):
    return scale * torch.randn(tensor.shape).to(device)


ACTIVATION_DICT = {'relu': nn.ReLU(), 'none': lambda x: x}


class Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        input_dim, architecture, hidden_activation, output_activation = model_config["input_dim"], model_config[
            "architecture"], model_config["hidden_activation"], model_config["output_activation"]
        assert hidden_activation in ACTIVATION_DICT
        assert output_activation in ACTIVATION_DICT
        self.hidden_act = ACTIVATION_DICT[hidden_activation]
        self.output_act = ACTIVATION_DICT[output_activation]

        cnn_config, linear_config, split_config = Model._separate_config(architecture)
        self.cnn_layers, output_dim = Model._initialize_cnn_layers(input_dim, cnn_config, self.hidden_act)
        self.linear_layers, output_dim = Model._initialize_linear_layers(output_dim, linear_config, self.hidden_act)
        self.left_layers, self.right_layers = Model._initialize_split_layers(output_dim, split_config, self.hidden_act)

    @staticmethod
    def _separate_config(model_config):
        cnn_config, linear_config, split_config = [], [], []
        layer_type = "conv"

        for layer_config in model_config:
            layer_name = layer_config["name"].lower()
            if "conv" in layer_name:
                assert layer_type == "conv", "Conv layer configuration cannot be parsed correctly"
                cnn_config.append(layer_config)
            elif "linear" in layer_name:
                assert layer_type in ["conv", "linear"], "Linear layer configuration cannot be parsed correctly"
                layer_type = "linear"
                linear_config.append(layer_config)
            elif "split" in layer_name:
                assert layer_type in ["conv", "linear", "split"], "Split layer configuration cannot be parsed correctly"
                layer_type = "split"
                split_config.append(layer_config)
            else:
                "Model layer cannot be parsed correctly"
        return cnn_config, linear_config, split_config

    @staticmethod
    def _initialize_cnn_layers(input_dim, cnn_config, hidden_activation):
        cnn_layers = nn.ModuleList([])
        for i in range(len(cnn_config)):
            layer_dict = cnn_config[i]

            in_channels = input_dim[0] if i == 0 else cnn_config[i - 1]['channels']
            out_channels = layer_dict['channels'] if 'channels' in layer_dict else None
            kernel_size = layer_dict['kernel_size'] if 'kernel_size' in layer_dict else 4
            stride = layer_dict['stride'] if 'stride' in layer_dict else 1
            padding = layer_dict['padding'] if 'padding' in layer_dict else 0

            new_height = int((input_dim[1] - kernel_size + 2 * padding) / stride) + 1
            new_width = int((input_dim[2] - kernel_size + 2 * padding) / stride) + 1
            input_dim = [out_channels, new_height, new_width]

            cnn_layers.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride))
            if i != len(cnn_config) - 1:
                cnn_layers.append(hidden_activation)

        mult = 1
        for val in input_dim:
            mult *= val
        return cnn_layers, [mult]

    @staticmethod
    def _initialize_linear_layers(input_dim, linear_config, hidden_activation):
        linear_layers = nn.ModuleList([])
        for i in range(len(linear_config)):
            layer_dict = linear_config[i]

            in_size = input_dim[0] if i == 0 else linear_config[i - 1]['size']
            out_size = layer_dict['size'] if 'size' in layer_dict else None
            assert in_size is not None, "Size is none for linear layer"
            assert out_size is not None, "Size is none for linear layer"

            linear_layers.append(nn.Linear(in_size, out_size))
            if i != len(linear_config) - 1:
                linear_layers.append(hidden_activation)
        return linear_layers, [linear_config[-1]['size']] if len(linear_config) != 0 else input_dim

    @staticmethod
    def _initialize_split_layers(input_dim, split_config, hidden_activation):
        left_layers, right_layers = nn.ModuleList([]), nn.ModuleList([])
        for i in range(len(split_config)):
            layer_dict = split_config[i]

            in_sizes = [input_dim[0]] * len(layer_dict['sizes']) if i == 0 else split_config[i - 1]['sizes']
            out_sizes = layer_dict['sizes'] if 'sizes' in layer_dict else None
            assert in_sizes is not None, "Size is none for linear layer"
            assert out_sizes is not None, "Size is none for linear layer"

            left_layers.append(nn.Linear(in_sizes[0], out_sizes[0]))
            right_layers.append(nn.Linear(in_sizes[1], out_sizes[1]))
            if i != len(split_config) - 1:
                left_layers.append(hidden_activation)
                right_layers.append(hidden_activation)
        return left_layers, right_layers

    def forward(self, x):
        if len(self.cnn_layers) != 0:
            for layer in self.cnn_layers:
                x = layer(x)
            x = torch.flatten(x, start_dim=1)
            if len(self.linear_layers) != 0 or len(self.left_layers) != 0 and len(self.right_layers) != 0:
                x = self.hidden_act(x)

        if len(self.linear_layers) != 0:
            for layer in self.linear_layers:
                x = layer(x)
            if len(self.left_layers) != 0 and len(self.right_layers) != 0:
                x = self.hidden_act(x)

        if len(self.left_layers) != 0 and len(self.right_layers) != 0:
            l, r = x, x
            for layer in self.left_layers:
                l = layer(l)
            for layer in self.right_layers:
                r = layer(r)
            x = [l, r]

        if len(x) == 2:
            return self.output_act(x[0]), self.output_act(x[1])
        return self.output_act(x)
