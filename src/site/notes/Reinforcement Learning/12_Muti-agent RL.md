---
{"dg-publish":true,"permalink":"/reinforcement-learning/12-muti-agent-rl/","dgPassFrontmatter":true,"created":"2023-10-20T15:22:25.590+08:00"}
---

代码  [19\_多智能体.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/19_%E5%A4%9A%E6%99%BA%E8%83%BD%E4%BD%93.ipynb)

# 1. 多智能体基础

多智能体条件下，环境是非稳态的，每个智能体都在改变环境，所以转移概率也可能经常变化。

* 完全中心化（fully centralized）方法：将多个智能体进行决策当作一个超级智能体在进行决策，即把所有智能体的状态聚合在一起当作一个全局的超级状态，把所有智能体的动作连起来作为一个联合动作，好处是环境仍然是稳态的，收敛性有保证，但是状态空间或者动作空间太大可能导致维度爆炸。

* 完全去中心化（fully decentralized）：假设每个智能体都在自身的环境中独立地进行学习，不考虑其他智能体的改变。每个智能体单独采用一个强化算法训练，但是环境非稳态，收敛性不能保证。

代码中仍然可以定义 PPO 或者其他算法，在训练时，建立多个智能体，每个智能体单独用一个 transition 表即可。

# 2. 中心化训练去中心化执行(CTDE)

这是指在训练的时候使用一些单个智能体看不到的全局信息而以达到更好的训练效果，而在执行时不使用这些信息，每个智能体完全根据自己的策略直接动作以达到去中心化执行的效果。

中心化训练去中心化执行的算法能够在训练时有效地利用全局信息以达到更好且更稳定的训练效果，同时在进行策略模型推断时可以仅利用局部信息，使得算法具有一定的扩展性。

CTDE 可以类比成一个足球队的训练和比赛过程：在训练时，11 个球员可以直接获得教练的指导从而完成球队的整体配合，而教练本身掌握着比赛全局信息，教练的指导也是从整支队、整场比赛的角度进行的；而训练好的 11 个球员在上场比赛时，则根据场上的实时情况直接做出决策，不再有教练的指导。

[Is MAPPO All You Need in Multi-Agent Reinforcement Learning?](https://d2jud02ci9yv69.cloudfront.net/2024-05-07-is-mappo-all-you-need-128/blog/is-mappo-all-you-need/) 一文表明，CTDE 的 MAPPO 方案可能不如 IPPO 及其改进方法。详细介绍参考 [[Reinforcement Learning/12_Muti-agent RL#6. MAPPO\| MAPPO一节]]。

# 3. 多智能体任务的状态设计

可以发现很多类似任务设计中，如果是比较复杂的状态，比如是数组的组合，复合数组等，都是“暴力”拼接、拉直的，类似 [[Machine Learning/MLP-ABC/06_其他机器学习技术#卷积神经网络\|CNN]] 中把最后提取的特征展平到 [[Machine Learning/MLP-ABC/02_神经网络\|MLP]] 一样，对人类来说这样做很难学到任何东西，但是对神经网络来说是有效的。

在 CDTE 框架中，critic 需要接收全局状态输入，如果有 n 个智能体，每个智能体的 state 长度是 m，那么全局状态维度是 n\*m.

# 4. 环境特征

类似 Gymnasium 的 PettingZoo 环境给出的 state 一般是一个字典，键是智能体名称，值是对应智能体的状态。输入的 actions 一般也是一个字典，键是智能体名称，值是动作。

```python
obv, info = env.reset()

obv = {'agent_1': array([...]),
	   'agent_2': array([...]),
	   ...}
	   
action = {"agent_1": action_1,
		  "agent_2": action_2,
		  ...}
		  
next_state, reward, done, truncated, info = env.step(action)
```
# 5. IDQN

给每个智能体单独配置一个 DQN 即可，经验池共用或单独配置需要视情况而定，取决于 agent 的 state 和 action 空间是否相同，任务是否相同等因素。

# 6. MADDPG


# 7. MAPPO

原生 MAPPO 中，由于智能体是同质的，因此只分配了两个网络，一个演员一个评论员，所有智能体共用一套参数。如果不同质，那么演员网络就很难用一个网络解决，每个 agent 单独配置一个 actor 网络；critic 仍然可以只用一个，接收全局状态输入，这也是 CTDE 方案。

网络方面，原生 MAPPO 采用的是 RNN 而不是普通的 RNN。

[Is MAPPO All You Need in Multi-Agent Reinforcement Learning?](https://d2jud02ci9yv69.cloudfront.net/2024-05-07-is-mappo-all-you-need-128/blog/is-mappo-all-you-need/) 一文中介绍了一些基于 IPPO 的改进方法，并且在星际争霸 II 中表现优于 MAPPO，下面介绍这几种方法。

## 7.1. IPPO

每个智能体单独配置一个 PPO 算法，每个 actor 和 critic 都只接收自己本地的信息，需要注意的是，后面的两个方法每个 agent 都单独配置了 critic，所以都属于 IPPO 的改进，也称 MAIPPO。

## 7.2. MAPPO-FP

对于值函数的输入，MAPPO-FP 将当前智能体自己的信息 own_feats（比如 `agent ID` 、 `position` `last action` 等）与全局信息 $s$ 连接起来。

$$V_i(s)=V^{\boldsymbol{\phi}}(\mathrm{concat}(s,f^i))$$

## 7.3. Noisy-MAPPO

也是改值函数输入，$s$ 是全局状态，$x^i$ 是高斯分布的噪音向量，直接 concat 到 state 后面，避免过拟合。实验表明 Noisy-MAPPO 效果最好。

$$V_i(s)=V^{\boldsymbol{\phi}}(\mathrm{concat}(s,x^i)),\quad x^i\sim\mathcal{N}(0,\sigma^2I)$$