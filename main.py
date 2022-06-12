# %% import packages
import torch
import numpy as np
import gym_super_mario_bros

from pathlib import Path
from datetime import datetime
from nes_py.wrappers import JoypadSpace

from Metric import MetricLogger
from MarioNet import MarioNet
from Wrappers import wrappers
from Mario import Mario

# making environment
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

# 由于算力限制，这里将 action 限制在向右走以及向右跳之间
# 因此, 此时的 Action space 是 Discrete(2)
env = JoypadSpace(env, [['right'], ['right', 'A']])
env = wrappers(env)  # 套上 wrapper, 让数据相对容易处理些

# getting ready for Logger
save_dir = Path('checkpoints') / datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
save_dir.mkdir(parents=True)
logger = MetricLogger(save_dir)

# %% Training Model
mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

episodes = 100  # 训练次数

for e in range(episodes):
    state = env.reset()  # 在每个 Episode 之前 reset 环境
    # 开一局游戏
    while True:
        action = mario.act(state)  # 让 Mario 根据当前的 State 选择动作
        next_state, reward, done, info = env.step(action)  # 对 environment 执行动作, 也就是按下对应的按键
        mario.cache(state, next_state, action, reward, done)  # 记住当前的 status
        q, loss = mario.learn()  # Learning
        state = next_state  # 将 State 更新为下一个状态
        logger.log_step(reward, loss, q)  # 记录下 Training 的一部分 Metrics
        if done or info['flag_get']:  # 当 game_over 或者拿到最终的 flag 的时候, 游戏结束
            break

    logger.log_episode()

    # 每 20 个 Episode 输出信息到 Console 上
    if e % 20 == 0:
        logger.record(episode=e, epsilon=mario.exp_rate, step=mario.curr_step)
    # 每 200 个 Episode 保存为 checkpoint 到对应的文件夹
    if e % 200 == 0:
        mario.save()

mario.save()  # 在 Training 的最后保存模型
# %% Testing Model

load_path = './models/mario_net.chkpt'
checkpoint = torch.load(load_path)
network = MarioNet(input_dim=(4, 84, 84), output_dim=env.action_space.n)
network.load_state_dict(checkpoint['model'])

test_episodes = 5

for e in range(test_episodes):
    state = env.reset()  # 在每一个 episode 开始之前都需要 reset environment
    rewards = []
    eps_reward = 0
    while True:
        # 将 state 转化为 network 输入需要的格式
        state = state.__array__().copy()
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        # 将 state 输入 MarioNet 中进行 predict
        action_values = network(state, 'online')
        action = torch.argmax(action_values).item()  # 输出的是概率最大的 action
        # 将得到的 action 作用在环境上
        next_state, reward, done, info = env.step(action)
        state = next_state
        # 渲染屏幕
        env.render()
        eps_reward += reward
        # 判定是否需要退出
        if done or info['flag_get']:
            rewards.append(eps_reward)
            break

env.close()  # 调用 close 方法关闭 GUI

print(f'{test_episodes} episodes\' mean rewards : {np.mean(rewards)}')
