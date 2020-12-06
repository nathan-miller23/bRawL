import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.callbacks import DefaultCallbacks

import gym
from models import *
import melee
from melee import SSBMEnv

import os, copy, dill
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
    return SSBMEnv(**env_config)

def get_trainer_from_params(params):
    return PPOTrainer(env="melee", config=params['rllib_params'])

#Smash env creator function

@ex.config
def my_config():
    ### Model params ###

    # whether to use recurrence in ppo model
    use_lstm = False

    # Base model params
    NUM_HIDDEN_LAYERS = 4
    SIZE_HIDDEN_LAYERS = 128
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
    train_batch_size = 6000

    # size of minibatches we divide up each batch into before
    # performing gradient steps
    # sgd_minibatch_size = 10000 if not LOCAL_TESTING else 800
    sgd_minibatch_size = 1000

    # Rollout length
    rollout_fragment_length = 800
    
    # Whether all PPO agents should share the same policy network
    shared_policy = True

    # Number of training iterations to run
    num_training_iters = 2000 

    # Stepsize of SGD.
<<<<<<< HEAD
    lr = 5e-5
=======
    lr = 5e-4
>>>>>>> 6aae0384af665de412a0905047bf8dca92294601

    # Learning rate schedule.
    lr_schedule = None

    # If specified, clip the global norm of gradients by this amount
    grad_clip = 0.1

    # Discount factor
    gamma = 0.95

    # Exponential decay factor for GAE (how much weight to put on monte carlo samples)
    # Reference: https://arxiv.org/pdf/1506.02438.pdf
    lmbda = 0.95

    # Whether the value function shares layers with the policy model
    vf_share_layers = False

    # How much the loss of the value network is weighted in overall loss
<<<<<<< HEAD
    vf_loss_coeff = 2e-2

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 0.02
    entropy_coeff_end = 0.00005
    entropy_coeff_horizon = 2e6
=======
    vf_loss_coeff = 1e-2

    # Entropy bonus coefficient, will anneal linearly from _start to _end over _horizon steps
    entropy_coeff_start = 5e-3
    entropy_coeff_end = 1e-5
    entropy_coeff_horizon = 1e6
>>>>>>> 6aae0384af665de412a0905047bf8dca92294601

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
<<<<<<< HEAD
    dolphin_exe_path = "/Users/chevin/Desktop/Launchpad/bRawL/mocker/dolphin-emu.app/Contents/MacOS"
    ssbm_iso_path = "/Users/chevin/Desktop/Launchpad/SSBMISO/SSMB.iso"
=======
    dolphin_exe_path = "/Applications/dolphin-emu.app/Contents/MacOS"
    ssbm_iso_path = "/Users/nathan/games/melee/SSMB.iso"
>>>>>>> 6aae0384af665de412a0905047bf8dca92294601
    char1 = melee.Character.CPTFALCON
    char2 = melee.Character.MARTH
    stage = melee.Stage.FINAL_DESTINATION
    cpu = True
    cpu_level = 2
    log = False
    reward_func = None
    render = False
    aggro_coeff = 1.0
    shaping_coeff = 1.0
    off_stage_weight = 10

    environment_params = {
        "dolphin_exe_path": dolphin_exe_path,
        "ssbm_iso_path": ssbm_iso_path,
        "char1": char1,
        "char2": char2,
        "stage": stage,
        "cpu": cpu,
        "cpu_level": cpu_level,
        "log": log,
        "reward_func": reward_func,
        "render": render,
        "aggro_coeff" : aggro_coeff,
        "shaping_coeff" : shaping_coeff,
        "off_stage_weight" : off_stage_weight,
        "gamma" : gamma
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
            "model" : {"custom_model_config": model_params, "custom_model": "my_model"},
<<<<<<< HEAD
            "callbacks": {"on_train_result": on_train_result}
=======
            "callbacks" : TrainingCallbacks
>>>>>>> 6aae0384af665de412a0905047bf8dca92294601
        }
    }

def increment_cpu_level(env):
<<<<<<< HEAD
    if env.cpu_level < 10:
        env.cpu_level += 1
        print("New cpu level: ", env.cpu_level)

def on_train_result(info):
    result = info["result"]
    if result["episode_reward_mean"] > 200:
        trainer = info["trainer"]
        trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: increment_cpu_level(env)))
=======
    env.cpu_level = max(env.cpu_level + 2, 9)

class TrainingCallbacks(DefaultCallbacks):

    def on_train_result(self, trainer, result, **kwargs):
        my_kills = result['custom_metrics']["KOs_ai_1_mean"]
        if False:
            trainer.workers.foreach_worker(
                lambda ev: ev.foreach_env(
                    lambda env: increment_cpu_level(env)))

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
            """
            Used in order to add custom metrics to our tensorboard data

            sparse_reward (int) - total reward from deliveries agent earned this episode
            shaped_reward (int) - total reward shaping reward the agent earned this episode
            """
            # Get SSBMEnv.py env from rllib wrapper
            env = base_env.get_unwrapped()[0]

            # List of episode stats we'd like to collect by agent
            stats_to_collect = env.vals_to_log

            # Store per-agent game stats to rllib info dicts
            for stat in stats_to_collect:
                for agent in env.agents:
                    info_dict = episode.last_info_for(agent)
                    episode.custom_metrics[stat + "_" + agent] = info_dict[stat]

>>>>>>> 6aae0384af665de412a0905047bf8dca92294601

def save_trainer(trainer, params, path=None):
    """
    Saves a serialized trainer checkpoint at `path`. If none provided, the default path is
    ~/ray_results/<experiment_results_dir>/checkpoint_<i>/checkpoint-<i>
    Note that `params` should follow the same schema as the dict passed into `gen_trainer_from_params`
    """
    # Save trainer
    save_path = trainer.save(path)

    # Save params used to create trainer in /path/to/checkpoint_dir/config.pkl
    config = copy.deepcopy(params)
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")

    # Note that we use dill (not pickle) here because it supports function serialization
    with open(config_path, "wb") as f:
        dill.dump(config, f)
    return save_path

def load_trainer(save_path):
    """
    Returns a ray compatible trainer object that was previously saved at `save_path` by a call to `save_trainer`
    Note that `save_path` is the full path to the checkpoint FILE, not the checkpoint directory
    """
    # Read in params used to create trainer
    config_path = os.path.join(os.path.dirname(save_path), "config.pkl")
    with open(config_path, "rb") as f:
        # We use dill (instead of pickle) here because we must deserialize functions
        config = dill.load(f)
    
    # Override this param to lower overhead in trainer creation
    config['training_params']['num_workers'] = 0

    # Get un-trained trainer object with proper config
    trainer = gen_trainer_from_params(config)

    # Load weights into dummy object
    trainer.restore(save_path)
    return trainer

@ex.automain
def main(params):
    ray.init()
    print(LOCAL_TESTING)
    ModelCatalog.register_custom_model("my_model", RllibPPOModel)
    register_env("melee", _env_creator)
    trainer = get_trainer_from_params(params)
    print("Trainer built")
    for i in range(params['num_training_iters']):
        result = trainer.train()
        print("Iteration {}".format(i))
        print("Reward: {}", result['episode_reward_mean'])
        if (i % 5 == 0) or (i == params['num_training_iters'] - 1):
            save_trainer(trainer, params)
