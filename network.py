import torch
import torch.nn as nn

class Qnet(nn.Module):
  def __init__(self, s_dim, a_dim):
    super().__init__()
    
    self.act = nn.ReLU()
    self.layer1 = nn.Linear(s_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.adv = nn.Linear(64, a_dim)
    self.val = nn.Linear(64, a_dim)
    
  def forward(self, s):
    x = self.act(self.layer1(s))
    x = self.act(self.layer2(x))

    adv = self.adv(x)
    val = self.val(x)
    q = val + adv - adv.mean(1).reshape(-1, 1)
    return q

  