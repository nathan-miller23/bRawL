# Library Imports
import os
import torch
import ray
import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer

# Local Imports
from environments import *
from models import *
from libmelee import SSBMEnv

# Create an experiment object
from sacred import Experiment
ex = Experiment("My Experiment", interactive=True)

# Necessary work-around to make sacred pickling compatible with rllib
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Whether we're running on the server or not
LOCAL_TESTING = os.getenv('RUN_ENV', 'local') == 'local'

# Sacred Slack Observer
from sacred.observers import SlackObserver
if os.path.exists('slack.json') and not LOCAL_TESTING:
    slack_obs = SlackObserver.from_config('slack.json')
    ex.observers.append(slack_obs)
    # Necessary for capturing stdout in multiprocessing setting
    SETTINGS.CAPTURE_MODE = 'sys'
    
def get_env_creator(name):
    def _create_env(env_config):
        return BreakoutRllib()
    return _create_env

def get_trainer_from_params(params):
    return PPOTrainer(env=params['env_name'], config=params['rllib_params'])

@ex.config
def config():
    # Define any parameters and its defaults here
    
    use_gym_env = True
    env_name = "Breakout-v0"
    use_gpu = False
    model = "PPO"
    framework = "torch"
    
    checkpoint_path = ""
    num_training_iters = 10
    num_workers = 1
    lr = 0.001
    train = True
    
    # Environment. Gym or your own.
    # Type of algorithm
    # Specifications of algorithm
    # GPU vs CPU
    # Torch vs Tensorflow - tf, tfe, or torch
    params = {
        "use_gym_env" : use_gym_env,
        "env_name" : env_name,
        
        "use_gpu" : use_gpu,
        "model" : model,
        "framework": framework,
        'num_training_iters': num_training_iters,
        
        "checkpoint_path": checkpoint_path,
        "train": train,
        
        'rllib_params': {
            "model": {
                "custom_model": "my_model",
            },
            "vf_share_layers": True,
            "lr": lr,
            "num_workers": num_workers,  # parallelism
            "framework": "torch"
        },
    }
    
@ex.main
def main(params):    
    for (key, value) in params.items():
        print("Parameter {} is set to {}".format(key, value))
        
    if not params["use_gym_env"]:
        register_env(params["env_name"], get_env_creator(params["env_name"]))
        
    if params["model"] == "DQN":
        print(3)
        from ray.rllib.agents import dqn
        ray.init()
        print(4)
        
        config = dqn.DEFAULT_CONFIG.copy()
        config["framework"] = params["framework"]
        env = str(params["env_name"])
        print(5)
        
        trainer = dqn.DQNTrainer(config=config, env=env)
        print(6)
        for i in range(100):
            print(trainer.train()['episode_reward_mean'])
    if params["model"] == "PPO":
        ray.init()
        ModelCatalog.register_custom_model("my_model", TorchCustomModel)
        trainer = get_trainer_from_params(params)

        if params["train"]:
            for i in range(params['num_training_iters']):
                print("starting training iteration {}".format(i))
                trainer.train()
                if i == params['num_training_iters'] - 1:
                    checkpoint_path = trainer.save()
                    print(checkpoint_path)
#     else:
#     trainer.restore(checkpoint_path)
#     trainer.restore("/root/ray_results/PPO_my_env_2020-11-01_19-04-39dkp16k9z/checkpoint_10/checkpoint-10")
#     env = _env_creator(params['rllib_params']['env_config'])
#     observation = env.reset()
#     for i in range(100):
#         plt.imshow(env.render(mode='rgb_array'))
#         display.display(plt.gcf())
#         display.clear_output(wait=True)
#         action = trainer.compute_action(observation)
#         observation, reward, done, info = env.step(action)
#         print("Reward: {}".format(reward))
#         if done: 
#             print("Episode finished after {} timesteps".format(i+1))
#             break
#     env.close()

if __name__ == '__main__':
    ex.run_commandline()