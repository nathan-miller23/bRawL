# Begin with some imports
import torch
import ray
import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
from environments import BreakoutRllib

# Create an experiment object
from sacred import Experiment
ex = Experiment("My Experiment", interactive=True)

@ex.config
def dqn_config():
    # Define any parameters and its defaults here
    
    use_gym_env = True
    env_name = "Breakout-v0"
    use_gpu = False
    model = "DQN"
    framework = "torch"
    
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
    }
@ex.main
def main(params):
    for (key, value) in params.items():
        print("Parameter {} is set to {}".format(key, value))
        
    if not params["use_gym_env"]:
        print(1)
        from ray.tune.registry import register_env
        print(2)
        env_class = params["env_name"]+"()"
        obj = eval(env_class)
        register_env(params["env_name"], lambda _: obj)
        
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

if __name__ == '__main__':
    ex.run_commandline()