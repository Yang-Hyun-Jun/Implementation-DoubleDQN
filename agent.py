import torch
import random
import numpy as np 
import torch.nn as nn 
from network import Qnet

class Agent:
  def __init__(self, s_dim, a_dim, lr, tau, gamma):
    self.qnet = Qnet(s_dim, a_dim)
    self.qnet_target = Qnet(s_dim, a_dim)
    self.qnet_target.load_state_dict(self.qnet.state_dict())

    self.s_dim = s_dim
    self.a_dim = a_dim
    self.gamma = gamma
    self.tau = tau
    self.eps = 1.0

    self.huber = nn.SmoothL1Loss()
    self.optim = torch.optim.Adam(self.qnet.parameters(), lr=lr)

  def get_action(self, s):
    with torch.no_grad():
      q = self.qnet(s)

      if np.random.uniform(0, 1, 1) <= self.eps:
        action = np.array(random.choice(range(self.a_dim)), dtype=int)
      else:
        action = np.array(q.argmax(dim=-1), dtype=int)
      
      return action

  def update(self, s, a, r, ns, done):
    """
    Double DQN update
    """
    with torch.no_grad():
      max_action = self.qnet(ns).argmax(dim=-1).view(-1, 1)
      next_q = self.qnet_target(ns).gather(1, max_action)
      q_target = r + self.gamma * next_q * (1-done)

    q = self.qnet(s).gather(1, a)
    q_loss = self.huber(q, q_target)
        
    self.optim.zero_grad()
    q_loss.backward()
    self.optim.step()
    self.soft_target_update()
    return q_loss.item()

  def soft_target_update(self):
    for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
        target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)