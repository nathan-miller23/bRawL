import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer

import gym
from models import *
import melee
from melee import SSBMEnv

import os
from sacred import Experiment
ex = Experiment("PPO Libmelee Training")

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
    return SSBMEnv(env_config["dolphin_exe_path"], env_config["ssbm_iso_path"], char1=env_config["char1"], char2=env_config["char2"], 
    stage=env_config["stage"], symmetric=env_config["symmetric"], cpu_level=env_config["cpu_level"], log=env_config["log"],
    reward_func=env_config["reward_func"], render=env_config["render"])

def get_trainer_from_params(params):
    return PPOTrainer(env="melee", config=params['rllib_params'])

#Smash env creator function

@ex.config
def my_config():
    ### Model params ###

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # LSTM memory cell size (only used if use_lstm=True)
    CELL_SIZE = 256

    ### Training Params ###

    num_workers = 1

    # list of all random seeds to use for experiments, used to reproduce results
    seeds = [0]

    # Placeholder for random for current trial
    seed = None

    # Number of gpus the central driver should use
    num_gpus = 0

    # How many environment timesteps will be simulated (across all environments)
    # for one set of gradient updates. Is divided equally across environments
    # train_batch_size = 40000 if not LOCAL_TESTING else 800
    train_batch_size = 4000

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    # sgd_minibatch_size = 10000 if not LOCAL_TESTING else 800
    sgd_minibatch_size = 128

    # Rollout length
    rollout_fragment_length = 200
    
    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 2000 

    # Stepsize of SGD.
    lr = 5e-3

    # Learning rate schedule.
    lr_schedule = None

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.99

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.98

    # Whether the value function shares layers with the policy model
    vf_share_layers = True

    # How much the loss of the value network is weighted in overall loss
    vf_loss_coeff = 1e-4

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.02
    entropy_coeff_end = 0.00005
    entropy_coeff_horizon = 3e5

    # Initial coefficient for KL divergence.
    kl_coeff = 0.2

    # PPO clipping factor
    clip_param = 0.05

    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    num_sgd_iter = 8 

    # To be passed into rl-lib model/custom_options config
    model_params = {
        "use_lstm" : use_lstm,
        "NUM_HIDDEN_LAYERS" : NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS" : SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS" : NUM_FILTERS,
        "NUM_CONV_LAYERS" : NUM_CONV_LAYERS,
        "CELL_SIZE" : CELL_SIZE,
        "HIDDEN_OUTPUT_SIZE": 4900
    }

    #Custom environment parameters
    dolphin_exe_path = "/Users/volk/Desktop/bRawL/dolphin-emu.app/Contents/MacOS"
    ssbm_iso_path = "/Users/volk/Downloads/SSMB.iso"
    char1 = melee.Character.FOX
    char2 = melee.Character.FALCO
    stage = melee.Stage.FINAL_DESTINATION
    symmetric = False
    cpu_level = 1
    log = False
    reward_func = None
    render = False

    environment_params = {
        "dolphin_exe_path": dolphin_exe_path,
        "ssbm_iso_path": ssbm_iso_path,
        "char1": char1,
        "char2": char2,
        "stage": stage,
        "symmetric": symmetric,
        "cpu_level": cpu_level,
        "log": log,
        "reward_func": reward_func,
        "render": render
    }

    params= {
        "num_training_iters": num_training_iters,
        "rllib_params": {
            "env_config": environment_params,
            "monitor": True,
            "framework": "torch",
        "preprocessor_pref":"deepmind",
        "num_workers" : num_workers,
        "train_batch_size" : train_batch_size,
        "sgd_minibatch_size" : sgd_minibatch_size,
        "rollout_fragment_length" : rollout_fragment_length,
        "num_sgd_iter" : num_sgd_iter,
        "lr" : lr,
        "lr_schedule" : lr_schedule,
        "grad_clip" : grad_clip,
        "gamma" : gamma,
        "lambda" : lmbda,
        "vf_share_layers" : vf_share_layers,
        "vf_loss_coeff" : vf_loss_coeff,
        "kl_coeff" : kl_coeff,
        "clip_param" : clip_param,
        "num_gpus" : num_gpus,
        "seed" : seed,
        "entropy_coeff_schedule" : [(0, entropy_coeff_start), (entropy_coeff_horizon, entropy_coeff_end)],
        },
        #"model" : {"custom_model_config": model_params, "custom_model": "my_model"}},
        "explore": True,
        "exploration_config":{
            "type":"EpsilonGreedy",
            "epsilon_timesteps": 100000
        }
    }

@ex.automain
def main(params):
    ray.init()
    print(LOCAL_TESTING)
    #ModelCatalog.register_custom_model("my_model", RllibPPOModel)
    register_env("melee", _env_creator)
    trainer = get_trainer_from_params(params)
    print("Trainer built")
    for i in range(params['num_training_iters']):
        result = trainer.train()
        print("Iteration {}".format(i))
        print("Reward: {}", result['episode_reward_mean'])
        if (i % 100 == 0) or (i == params['num_training_iters'] - 1):
            checkpoint_path = trainer.save()
            print(checkpoint_path)
