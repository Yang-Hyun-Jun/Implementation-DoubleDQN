# Implementation: Double DQN

My implementation code of Double DQN

[Deep Reinforcement Learning with Double Q-learning, 2016 AAAI](https://arxiv.org/pdf/1509.06461.pdf))

# Overview

- This paper addresses the issue of overestimation in DQN.
- It takes inspiration from the separation of Q for estimation and Q for behavior in traditional Double Q learning.
- Since DQN utilizes a target network, it leverages it to train Q in a Double Q learning style.
- However, the updating process of the target network follows the same approach as in DQN. (Soft or Hard target update)
- $$Y^{\text{DoubleDQN}}_t=R_{t+1}+\gamma Q_{\theta}(S_{t+1}, \argmax_aQ_{\theta'}(S_{t+1}, a))$$
