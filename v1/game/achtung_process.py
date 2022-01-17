import gym
import numpy as np
import matplotlib.pyplot as plt
from v1.game.achtung import Achtung
from v1.game.config import WINDOW_HEIGHT, WINDOW_WIDTH
class AchtungProcess(gym.Env):
    def __init__(self, n=1, frame_skip=1, obs_len=4, _id = 0, height = WINDOW_HEIGHT, width= WINDOW_WIDTH):
      self.env = Achtung(n,_id,speed=0,render_game=False, height=height, width=width)
      self.frame_skip = frame_skip
      self.obs_len = obs_len
      self.state = np.zeros((self.obs_len, self.env.window_width, self.env.window_height))
      self.action_space = gym.spaces.Discrete(3)
      self.observation_space = gym.spaces.Box(low=0, high=255,
        shape=(3, self.env.window_width, self.env.window_height), dtype=np.uint8)

    def step(self, action):
      _obs = []
      _reward = []
      done = False
      for t in range(self.frame_skip):
          obs, reward, done, info = self.env.step(action)
          _obs.append(obs)
          _reward.append(reward)

          if done:
            break

      obs_new = np.maximum(_obs[-1], _obs[-2] if len(_obs) > 1 else _obs[-1])
      self.state = obs_new

      return self.state, np.sum(_reward), done, {}   

    def reset(self):
      self.state = np.zeros((self.obs_len, self.env.window_width, self.env.window_height))
      obs = self.env.reset()
      self.state = obs
      
      return self.state

    def render(self):
      self.env.render()

