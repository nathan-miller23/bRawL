import gym
from environments.stacked import StackedEnv
from models.ppo import DiscretePPO

env = StackedEnv(gym.make('Breakout-v0'), 40, 40, 6, 1) #gym.make("CartPole-v1") #
env._max_episode_steps = 1400
state_dim = env.observation_space.shape
action_dim = env.action_space.n

# policy_config = {
#     "input_dim": list(state_dim),
#     "architecture": [{"name": "linear2", "size": 64},
#                      {"name": "linear2", "size": action_dim}],
#     "hidden_activation": "leaky_relu",
#     "output_activation": "none"
# }
# value_config = {
#     "input_dim": list(state_dim),
#     "architecture": [{"name": "linear1", "size": 64},
#                      {"name": "linear2", "size": 1}],
#     "hidden_activation": "leaky_relu",
#     "output_activation": "none"
# }

policy_config = {
    "input_dim": list(state_dim),
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "mark_embedding", "new_input_dim": 64},
                     {"name": "linear2", "size": 64},
                     {"name": "linear2", "size": action_dim}],
    "hidden_activation": "leaky_relu",
    "output_activation": "none"
}
value_config = {
    "input_dim": list(state_dim),
    "architecture": [{"name": "conv1", "channels": 32, 'kernel_size': 8, 'stride': 4},
                     {"name": "conv2", "channels": 64, 'kernel_size': 4, 'stride': 2},
                     {"name": "conv3", "channels": 64, 'kernel_size': 3, 'stride': 1},
                     {"name": "mark_embedding", "new_input_dim": 64},
                     {"name": "linear2", "size": 64},
                     {"name": "linear2", "size": 1}],
    "hidden_activation": "leaky_relu",
    "output_activation": "none"
}
model = DiscretePPO(policy_config, value_config, env, "cuda", warmup_games=1)
# model.train(500)
# model.load_model("Ant-v2-SAC-400", "cuda")
model.train(400, deterministic=False)
model.save_model("Pong-PPO-400")
model.eval(100)