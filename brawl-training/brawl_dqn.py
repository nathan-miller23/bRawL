import torch
import ray
import gym
import ray.rllib.agents.dqn as dqn

import os, sys

#Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath('libmelee/SSBMEnv.py'))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)


from SSBMEnv import SSBMEnv
from ray.tune.registry import register_env

ray.init()

def env_creator(env_config):
    return SSBMEnv(env_config['dolphin_exe_path'], env_config['ssbm_iso_path'])

# DEBUGGING INFO
#   had to rename Contents/MacOS/Dolphin to dolphin-emu
#   had to mkdir Pipes in Contents/Resources/User



register_env("SSBM", env_creator)
trainer = dqn.DQNTrainer(env="SSBM", config = {
    "env_config": {'dolphin_exe_path': '/Users/jimwang/Desktop/FM-v5.9-Slippi-r18-Mac/Dolphin.app/Contents/MacOS','ssbm_iso_path': '/Users/jimwang/Desktop/launchpad/SSMB.iso'}
})


print(trainer.train())
