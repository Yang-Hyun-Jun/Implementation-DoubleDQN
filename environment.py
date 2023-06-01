import numpy as np

class Environment:
    PRICE_COLUMN = -1  

    def __init__(self, data=None):
      self.data = data
      self.observation = None
      self.idx = 0

    def reset(self):
      self.observation = None
      self.idx = 0
      state, _ = self.observe()
      return state

    def observe(self):
      self.observation = self.data.iloc[5*self.idx:5*(self.idx+1)]
      done = 0 if len(self.data) >= 5*(self.idx + 2) else 1
      done = np.array([done])
      self.idx += 1
      return self.observation.to_numpy().reshape(-1), done

    def step(self, action):
      """
      action 0: 매도
      action 1: 홀딩
      """

      now_price = self.get_price()
      next_state, done = self.observe()
      next_price = self.get_price()

      sign = -1 if next_price < now_price else 1
      
      if (sign < 0) & (action == 0):
        reward = np.array([1])
      
      if (sign < 0) & (action == 1):
        reward = np.array([0])
      
      if (sign > 0) & (action == 0):
        reward = np.array([-1])
      
      if (sign > 0) & (action == 1):
        reward = np.array([0.5])

      return next_state, reward, done

    def get_price(self):
      return self.observation.iloc[-1]['close']