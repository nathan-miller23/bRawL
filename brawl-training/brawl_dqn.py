import torch
import ray
import gym
import ray.rllib.agents.dqn as dqn
import pickle
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn
from ray.rllib.utils.annotations import override



import os, sys

#Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath('libmelee/melee/SSBMEnv.py'))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)


from melee.SSBMEnv import SSBMEnv
from ray.tune.registry import register_env

ray.init()

def env_creator(env_config):
    return SSBMEnv(env_config['dolphin_exe_path'], env_config['ssbm_iso_path'], cpu=True)

class FCNet(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        input_size = obs_space.shape[-1]
        self._hiddens = nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_outputs)
        )
        self._features = None
        self._num_outputs = num_outputs



    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        self._features = self._hiddens(obs).view(-1, self._num_outputs)
        return self._features, state



register_env("SSBM", env_creator)
trainer = dqn.DQNTrainer(env="SSBM", config = {
    "model": {
        "custom_model": FCNet,
    },
    "gamma": 0.999,
    "framework": "torch",
    "env_config": {'dolphin_exe_path': '/Users/jimwang/Desktop/launchpad/bRawL/mocker/dolphin-emu.app/Contents/MacOS','ssbm_iso_path': '/Users/jimwang/Desktop/launchpad/SSMB.iso'},
    "exploration_config": {"epsilon_timesteps": 3000000},
    "hiddens": [256, 256],
    "output": 'brawl-training/results',
    "monitor": True,
    "lr": 1e-5
})

policy = trainer.get_policy()
model = policy.q_model
print(model)

#trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-12-04_05-02-006b64djc4/checkpoint_411/checkpoint-411')
for i in range(5000):
    info = trainer.train()
    print("episode reward mean:", info['episode_reward_mean'], "num steps trained:", info['info']["num_steps_trained"])
    if (i % 10 == 0):
        checkpoint = trainer.save()
        print('checkpoint saved at:', checkpoint)
        with open('brawl-training/saved-weights.pkl', 'wb') as file:
            weights = trainer.get_policy().get_weights()
            pickle.dump(weights, file)
