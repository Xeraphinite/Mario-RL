import torch
import numpy as np
import random
from collections import deque

from MarioNet import MarioNet

# 对 state tensor 化
def state_preprocess(state):
    state = state.__array__()
    state = torch.tensor(state).cuda()
    state = state.unsqueeze(0)
    return state


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.save_dir = save_dir

        self.net = MarioNet(self.state_dim, self.action_dim).float()

        self.exp_rate = 1
        self.exp_rate_decay = 0.99999975
        self.min_exp_rate = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=5000)  # 由于设备限制，Memory Buffer
        self.batch_size = 32

        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        # Epsilon Greedy
        if np.random.rand() < self.exp_rate:
            action_idx = np.random.randint(self.action_dim)  # 探索, 从 action space 中随机取样
        else:
            state = state_preprocess(state) # 将当前 state 进行预处理, 以让
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()  # 利用, 取出当前 model 觉得最好的 action
        # 降低当前的 exploration rate
        self.exp_rate = max(self.min_exp_rate, self.exp_rate * self.exp_rate_decay)
        self.curr_step += 1  # 增加当前的 step

        return action_idx

    # 保存 Memory
    def cache(self, state, next_state, action, reward, done):
        state = state_preprocess(state)
        next_state = state_preprocess(next_state)

        action = torch.tensor([action]).cuda()
        reward = torch.tensor([reward]).cuda()
        done = torch.tensor([done]).cuda()

        self.memory.append((state, next_state, action, reward, done,))

    # 从 Memory 中随机采样
    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # Learning
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    # Training
    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        state, next_state, action, reward, done = self.recall()  # 从 memory 中随机采样
        td_est = self.td_estimate(state, action)  # 取得 td_estimate
        td_tgt = self.td_target(reward, next_state, done)  # 取得 td_target
        loss = self.update_Q_online(td_est, td_tgt)  # 反向传播 loss

        return td_est.mean().item(), loss

    # 保存模型
    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exp_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")