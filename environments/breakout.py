import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class BreakoutRllib(gym.Env):
    
    def __init__(self, *args, **kwargs):
        self._env = gym.make('Breakout-v0')
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        
        
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, reward, done, info
    
    def reset(self):
        return self._env.reset()