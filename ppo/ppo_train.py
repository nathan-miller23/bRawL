import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ppo_model import *
import gym

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

@ex.config
def config():
    params = {
        "env": "my_env",
        'rllib_params': {
            "env_config": {
                "name": 'Breakout-v0',
            },
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": 0.001,
            "num_workers": 1,  # parallelism
            "framework": "torch"
        },
        'num_training_iters': 1
    }

@ex.automain
def main(params):
    ray.init()
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)
    register_env("my_env", _env_creator)

    trainer = get_trainer_from_params(params)

    for i in range(params['num_training_iters']):
        print("starting training iteration {}".format(i))
        trainer.train()
