import torch
import ray
import gym
import ray.rllib.agents.dqn as dqn

import os, sys

#Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath('libmelee/melee/SSBMEnv.py'))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)


from melee.SSBMEnv import SSBMEnv
from ray.tune.registry import register_env

ray.init()

def env_creator(env_config):
    return SSBMEnv(env_config['dolphin_exe_path'], env_config['ssbm_iso_path'])



register_env("SSBM", env_creator)
trainer = dqn.DQNTrainer(env="SSBM", config = {
    "framework": "torch",
    "env_config": {'dolphin_exe_path': '/Users/jimwang/Desktop/launchpad/bRawL/mocker/dolphin-emu.app/Contents/MacOS','ssbm_iso_path': '/Users/jimwang/Desktop/launchpad/SSMB.iso'},
    "exploration_config": {"epsilon_timesteps": 5000000},
    "hiddens": [256, 256, 256]
})

trainer.restore('/Users/jimwang/ray_results/DQN_SSBM_2020-11-28_23-42-54ed1wjkkl/checkpoint_1092/checkpoint-1092')
for i in range(5000):
    info = trainer.train()
    print("episode reward mean:", info['episode_reward_mean'], "num steps trained:", info['info']["num_steps_trained"])
    if (i % 10 == 0):
        checkpoint = trainer.save()
        print('checkpoint saved at:', checkpoint)
