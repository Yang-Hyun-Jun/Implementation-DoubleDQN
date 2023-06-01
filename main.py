import torch
import pandas as pd
import matplotlib.pyplot as plt

from environment import Environment
from buffer import ReplayMemory
from agent import Agent

def tensorize(array):
  array = array.reshape(1,-1)
  tensor = torch.tensor(array)
  return tensor

def make_batch(sample):
    x = list(zip(*sample))
    x = list(map(torch.cat, x))
    return x


if __name__ == "__main__":

    data = pd.read_csv('data/data.csv', index_col=0)

    s_dim = 5 * data.shape[1]
    a_dim = 2
    lr = 1e-4
    tau = 0.005
    norm = 1000
    gamma = 0.99
    episode = 500
    batch_size = 64
    buffer_size = int(5e4)
    eps_min = 0.1
    cums = []

    train_data = data.iloc[:200]
    test_data = data.iloc[200:]

    env = Environment(train_data)
    buffer = ReplayMemory(buffer_size)
    agent = Agent(s_dim, a_dim, lr, tau, gamma)

    for epi in range(episode):
       state = env.reset() / norm
       cum_r = 0
       steps = 0
       loss = None 

       while True:
            agent.eps = max(eps_min, agent.eps * 0.9999)
            action = agent.get_action(tensorize(state).float())
            next_state, reward, done = env.step(action)
            next_state = next_state / norm

            sample = (tensorize(state).float(),
                        tensorize(action),
                        tensorize(reward).float(),
                        tensorize(next_state).float(),
                        tensorize(done).float())
          
            steps += 1
            cum_r += reward[0]
            cums.append(cum_r)
            state = next_state
            buffer.push(sample)

            if len(buffer) >= batch_size:
                batch_data = buffer.sample(batch_size)
                batch_data = make_batch(batch_data)
                loss = agent.update(*batch_data)

            if done:
                print(f'epi:{epi}')
                print(f'eps:{agent.eps}')
                print(f'cumr:{cum_r}')
                print(f'loss:{loss}\n')
                break

    average_return = pd.DataFrame(cums).rolling(500).mean().dropna()[0]
    plt.figure(figsize=(20, 10))
    plt.rc('font', size=20)
    plt.plot(average_return)
    plt.xlabel('Step')
    plt.ylabel('Average Return')
    plt.savefig('result')
          
          
    