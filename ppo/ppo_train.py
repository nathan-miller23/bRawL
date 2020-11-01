import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ppo_model import *
import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt

import os
from sacred import Experiment
ex = Experiment("Rllib Example")

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Whether we're running on the server or not
LOCAL_TESTING = os.getenv('RUN_ENV', 'local') == 'local'

from sacred.observers import SlackObserver
if os.path.exists('slack.json') and not LOCAL_TESTING:
    slack_obs = SlackObserver.from_config('slack.json')
    ex.observers.append(slack_obs)

    # Necessary for capturing stdout in multiprocessing setting
    SETTINGS.CAPTURE_MODE = 'sys'

def _env_creator(env_config):
    return gym.make(env_config['name'])

def get_trainer_from_params(params):
    return PPOTrainer(env=params['env'], config=params['rllib_params'])

#Smash env creator function

@ex.config
def config():
    checkpoint_path = ""
    num_training_iters = 10
    num_workers = 1
    lr = 0.001
    train = True
    params = {
        "checkpoint_path": checkpoint_path,
        "train": train,
        "env": "my_env",
        'rllib_params': {
            "env_config": {
                "name": 'Breakout-v0',
            },
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": lr,
            "num_workers": num_workers,  # parallelism
            "framework": "torch"
        },
        'num_training_iters': num_training_iters
    }

@ex.automain
def main(params):
    ray.init()
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    register_env("my_env", _env_creator)

    trainer = get_trainer_from_params(params)
    
    # if params["train"]:
    #     for i in range(params['num_training_iters']):
    #         print("starting training iteration {}".format(i))
    #         trainer.train()
    #         if i == params['num_training_iters'] - 1:
    #             checkpoint_path = trainer.save()
    #             print(checkpoint_path)
    # else:
    # trainer.restore(checkpoint_path)
    trainer.restore("/root/ray_results/PPO_my_env_2020-11-01_19-04-39dkp16k9z/checkpoint_10/checkpoint-10")
    env = _env_creator(params['rllib_params']['env_config'])
    observation = env.reset()
    for i in range(100):
        plt.imshow(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)
        action = trainer.compute_action(observation)
        observation, reward, done, info = env.step(action)
        print("Reward: {}".format(reward))
        if done: 
            print("Episode finished after {} timesteps".format(i+1))
            break
    env.close()
