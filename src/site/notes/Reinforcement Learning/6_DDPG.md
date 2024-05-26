---
{"dg-publish":true,"permalink":"/reinforcement-learning/6-ddpg/","dgPassFrontmatter":true,"created":"2023-08-07T17:31:15.584+08:00"}
---

代码 [13\_DDPG.ipynb](https://github.com/Aegis1863/ML_practice/blob/master/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/13_DDPG.ipynb)

#机器学习/强化学习/异策略 #机器学习/强化学习/连续动作 #机器学习/强化学习/确定性策略

DDPG是DQN的改进版，很重要的一点就是把DQN拓展到连续动作空间，DQN只需要向Q网络输入状态，但是DDPG中需要输入状态和动作，因为在连续动作中，动作的取值是无限的，因此就不能输出每个动作的Q值再选动作，而是创建一个[[Reinforcement Learning/2_Policy Gradient\|策略网络]]作为演员，选择动作，与状态一起输入[[Reinforcement Learning/1_DQN\|Q网络]]，也就是评论员，Q网络只是评价这个状态动作对的价值。

- DDPG 引入软更新概念，即对于目标网络不再是延迟更新，而是每次只更新一点点，也就是说，不是把目标网络的参数直接更新为当前网络参数，而是按比例更新，比如下面伪代码;

```python
0.95 * TargetNet_params + 0.05 * Net_params
```
{ #5661e0}


#机器学习/强化学习/软更新 

- 把评论员网络变成Q网络，**同时输入状态和动作**，将它们拼接起来送入网络，之前DQN中没有这样做过，换成Q网络之后，输出一个价值，也就是这个动作状态的价值;

- 策略/演员网络直接输出动作，也就是确定性深度策略梯度中**确定性**的来源，之前是输出一个概率，现在是直接确定动作。

