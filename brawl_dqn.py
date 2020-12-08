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

#dir_path = os.path.dirname(os.path.realpath('libmelee/melee/SSBMEnv.py'))

#parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

#sys.path.insert(0, parent_dir_path)

from models import *
import melee
from melee.SSBMEnv import SSBMEnv
from ray.tune.registry import register_env

ray.init()

def env_creator(env_config):
    return SSBMEnv(**env_config)

model_params = {
    "NUM_HIDDEN_LAYERS" : 0,
    "SIZE_HIDDEN_LAYERS" : 256,
    "NUM_FILTERS" : 64,
    "NUM_CONV_LAYERS" : 3
}

register_env("SSBM", env_creator)
trainer = dqn.DQNTrainer(env="SSBM", config = {
    "model": {
        "custom_model_config": model_params,
        "custom_model": RllibDQNModel
    },
    "gamma": 0.995,
    "framework": "torch",
    "env_config": {
        'dolphin_exe_path': '/Users/jimwang/Desktop/launchpad/bRawL/mocker/dolphin-emu.app/Contents/MacOS',
        'ssbm_iso_path': '/Users/jimwang/Desktop/launchpad/SSMB.iso',
        "char1": melee.Character.KIRBY,
        "char2": melee.Character.MARTH,
        "cpu": True,
        "cpu_level": 3,
        'every_nth' : 1,
        'buffer_size' : 64,
        "gamma": 0.995
    },
    "hiddens": [256, 256],
    "output": 'brawl-training/results',
    "lr": 1e-4,
    "v_min": -300.0,
    "v_max": 300.0,
    "noisy" : True,
    "sigma0" : 0.2,
    "n_step" : 5,
    "exploration_config": {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "epsilon_timesteps": 200000
    }
})

policy = trainer.get_policy()
model = policy.q_model
print(model)

#TRAINED FOR ABOU 2 MILLION STEPS? USING LR 5E-4, WITH GAMMA FOR POTENTIAL trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-12-07_20-28-193m9o4_hj/checkpoint_353/checkpoint-353')
#LEVEL 1 CPU, lr 1e-5, NOISY   trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-12-08_02-18-051ylsgo_5/checkpoint_436/checkpoint-436')
#LEVEL 3 CPU, lr 1e-5, NO NOISY trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-12-08_02-24-41dt2u_cxj/checkpoint_171/checkpoint-171')
#LEVEL 3 CPU, NOISY, lr 3e-5 trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-12-08_03-42-140m30rkzi/checkpoint_363/checkpoint-363')
trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-12-08_10-22-49oyviw9x3/checkpoint_64/checkpoint-64')
for i in range(5000):
    info = trainer.train()
    print("episode reward mean:", info['episode_reward_mean'], "num steps trained:", info['info']["num_steps_trained"])
    if (i % 10 == 0):
        checkpoint = trainer.save()
        print('checkpoint saved at:', checkpoint)
        with open('brawl-training/saved-weights.pkl', 'wb') as file:
            weights = trainer.get_policy().get_weights()
            pickle.dump(weights, file)
