from melee import SSBMEnv
from ray.rllib.models import ModelCatalog
from dqn_train import load_trainer, _env_creator
from ray.tune.registry import register_env
from models import *
import numpy as np
import ray
import melee, time, os, argparse
from melee import Character
from models import bc
from ray.rllib.agents.dqn import dqn
from ppo_train import load_trainer
import dill
from ray.rllib.agents.ppo.ppo import PPOTrainer

str_to_char = {
    "fox" : Character.FOX,
    "falco" : Character.FALCO,
    "marth" : Character.MARTH,
    "kirby" : Character.KIRBY,
    "captain_falcon" : Character.CPTFALCON
}

def softmax(logits):
    e_x = np.exp(logits.T - np.max(logits))
    return (e_x / np.sum(e_x, axis=0)).T

class Policy():

    def __init__(self, *args, **kwargs):
        pass

    def action(self, observation):
        raise NotImplementedError("AHHH")

class PolicyFromRllib():
    def __init__(self, trainer):
        super().__init__()
        self.policy = trainer.get_policy('default_policy')

    def action(self, observation):
        logits = self.policy.compute_actions(np.expand_dims(observation, 0), None)[2]['action_dist_inputs']
        probs = np.squeeze(softmax(logits))
        action = np.random.choice(len(probs), p=probs)
        return action

class PolicyFromTorch():

    def __init__(self, path):
        if path == '/Users/jimwang/Desktop/yi_kirby_fox_yolo.jpg':
            self.agent = bc.LinearBufferAgent(buffer_len=4, num_states=87, num_actions=59, hidden_size=256)
        else:
            self.agent = bc.LinearBufferAgent(buffer_len=32, num_states=87, num_actions=69, hidden_size=256)
        self.agent.load_state(path)

    def action(self, observation):
        logits = self.agent(torch.FloatTensor(observation))
        return torch.distributions.categorical.Categorical(logits=logits).sample()

def evaluate(env, policy_1, policy_2):
    done = False
    joint_obs = env.reset()
    while not done:
        joint_action = {}
        if 'ai_1' in joint_obs:
            joint_action['ai_1'] = policy_1.action(joint_obs['ai_1'])
        if 'ai_2' in joint_obs:
            joint_action['ai_2'] = policy_2.action(joint_obs['ai_2'])
        joint_obs, _, done, _ = env.step(joint_action)
        done = done['__all__']

def jim_load(path, env_params, jim=True):
    ray.shutdown()
    ray.init()
    if not jim:
        # Read in params used to create trainer
        ModelCatalog.register_custom_model("my_model", RllibPPOModel)
        config_path = os.path.join(os.path.dirname(path), "config.pkl")
        with open(config_path, "rb") as f:
            # We use dill (instead of pickle) here because we must deserialize functions
            config = dill.load(f)
        config['rllib_params']['env_config'] = env_params
    else:
        model_params = {
            "NUM_HIDDEN_LAYERS" : 0,
            "SIZE_HIDDEN_LAYERS" : 256,
            "NUM_FILTERS" : 64,
            "NUM_CONV_LAYERS" : 3
        }
        config = {
            "model": {
                "custom_model_config": model_params,
                "custom_model": RllibDQNModel
            },
            "gamma": 0.995,
            "framework": "torch",
            "env_config": env_params,
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
        }


    def env_creator(env_config):
        return SSBMEnv(**env_config)
    register_env("SSBM", env_creator)
    register_env("melee", env_creator)

    if jim:
        trainer = dqn.DQNTrainer(env="SSBM", config=config)
    else:
        trainer = PPOTrainer(env="melee", config=config['rllib_params'])

    trainer.restore(path)
    return trainer

def get_policy(path, policy_type, env_params, jim=True):
    policy = None
    if policy_type == 'rllib':
        trainer = jim_load(path, env_params, jim)
        policy = PolicyFromRllib(trainer)
    elif policy_type == 'torch':
        policy = PolicyFromTorch(path)
    else:
        raise NotImplementedError("This type of policy is not supported")
    return policy

def get_policy_other(path, policy_type):
    policy = None
    if policy_type == 'rllib':
        trainer = load_trainer(path, env_params)
        policy = PolicyFromRllib(trainer)
    elif policy_type == 'torch':
        # TODO
        pass
    else:
        raise NotImplementedError("This type of policy is not supported")
    return policy

# PATH = "/Users/nathan/ray_results/DQN_melee_2020-12-07_16-50-5232lwbys_/checkpoint_96/checkpoint-96"

# register_env("melee", _env_creator)
# ModelCatalog.register_custom_model("my_model", RllibDQNModel)

# ray.init()
# trainer = load_trainer(PATH)
# trainer.workers.foreach_worker(
#                 lambda ev: ev.foreach_env(
#                     lambda env: env._stop_dolphin()))
# policy = trainer.get_policy('default_policy')
# ray.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dolphin_exe_path', '-e', default="/Applications/dolphin-emu.app/Contents/MacOS", help='The directory where dolphin is')
    parser.add_argument('--ssbm_iso_path', '-i', default="/Users/nathan/games/melee/SSMB.iso", help='Full path to Melee ISO file')
    parser.add_argument('--p1_path', '-pp1', required=True, type=str)
    parser.add_argument('--p2_path', '-pp2', required=True, type=str)
    parser.add_argument('--p1_type', '-tp1', default='rllib', type=str)
    parser.add_argument('--p2_type', '-tp2', default='rllib', type=str)
    parser.add_argument('--p1_frame_skip', '-fp1', default=1, type=int)
    parser.add_argument('--p2_frame_skip', '-fp2', default=1, type=int)
    parser.add_argument('--p1_buffer_size', '-bp1', default=64, type=int)
    parser.add_argument('--p2_buffer_size', '-bp2', default=64, type=int)
    parser.add_argument('--p1_character', '-cp1', choices=list(str_to_char.keys()), default='fox', type=str)
    parser.add_argument('--p2_character', '-cp2', choices=list(str_to_char.keys()), default='fox', type=str)
    args = parser.parse_args()

    env_params = {
        "dolphin_exe_path" : args.dolphin_exe_path,
        "ssbm_iso_path" : args.ssbm_iso_path,
        "buffer_size" : (args.p1_buffer_size, args.p2_buffer_size),
        "every_nth" : (args.p1_frame_skip, args.p2_frame_skip),
        "chars" : (str_to_char[args.p1_character], str_to_char[args.p2_character]),
        "same_char": args.p1_character == args.p2_character
    }

    policy_1 = get_policy(args.p1_path, args.p1_type, env_params, jim=False)
    policy_2 = get_policy(args.p2_path, args.p2_type, env_params, jim=True)
    env = SSBMEnv(**env_params)

    evaluate(env, policy_1, policy_2)
