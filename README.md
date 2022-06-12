# Mario-RL
## 项目简介

GDUFE - 大三下 - IT 技术创新实践课程项目

使用 **Double-DQN** 进行强化学习训练.

由于本人水平有限，希望大家不吝赐教！

## 项目架构

```
D:.
│   main.py
│   Mario.py
│   MarioNet.py
│   Metric.py
│   README.md
│   Wrappers.py
│
├───docs
│   │   RL.md
│   │
│   └───RL.assets
├───models
│       mario_net_2.chkpt
│
└───videos
        10000_episodes.mkv
        50000_episodes.mp4
```

直接运行 `main.py` 即可.

## 待改进

由于经济能力和设备的限制，同时又因为强化学习对设备算力极其严苛的要求，本次课程设计可能不会有太好的结果，未来如果能够有机会使用到更好的设备重新 train 的话效果可能会好很多. 同时，由于机器学习的可解释性问题，大部分的参数几乎都选择的是经验参数，好的参数可以将训练过程缩减、并让训练效果大大提升，当然，也需要长期的积累和持续的努力. 接下来从不同的方面说明当前实验需要改进的部分.

##### 训练结果相对较差

- 硬件角度
  - 更换更好的显卡、CPU
- 调整 environment
  - 采样、预处理的时候选择不同的方式
  - 重新设计 reward，将稀疏的 reward 尽量 dense 一些
- 调整 DQN 的角度
  - 将朴素的 Memory Replay 换成 Prioritized Replay
  - 使用 Dueling Network / 分布式 Q Function / Rainbow / … 调整强化学习过程
  - 调整当前 Double DQN 的超参数
  - 更换相对稳定的 DQN 实现，如 `stable-baseline` 中的 DQN 实现.
- 更换算法，使用 PPO 或 A3C 进行 Training

##### 可视化程度不足

在本次 Training 中，只使用了 matplotlib 进行可视化，未来可能考虑用 TensorBoard 实现可交互的可视化.

##### 没有写在 CPU 上 Training 的代码

为了简化思考以及 coding 的难度，本项目并没有写在 CPU 上 Train 的代码（因为要加很多判断 `cuda` 是否 `available` 的分支，比较麻烦），后续可能会将其补上.

## 参考资料

1. [Easy RL - 强化学习教程]: https://datawhalechina.github.io/easy-rl/#/

2. [Shusen Wang - 深度强化学习]: https://github.com/wangshusen/DeepLearning

3. [深度强化学习落地指南]:https://www.zhihu.com/column/c_1186982555915599872


## 代码参考

1. [PyTorch - Training a Mario-playing RL Agent]: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
