#### 项目简介



#### Super Mario 基本介绍

超级马力欧系列（日语：スーパーマリオ，英语：Super Mario）是任天堂开发，以旗下吉祥物——马力欧为主角的动作平台游戏系列，为《马力欧系列》中的核心作品，累计销量达 3 亿 1000 万份，为多个马力欧相关系列里最高，也帮助整个马力欧系列成为世界上销量最高的电子游戏系列。任天堂各主要家用机和掌上机上市后，平台上至少都有一部超级马力欧的新作。

在超级马力欧游戏中，玩家通常控制马力欧，在虚构的蘑菇王国中冒险。同行的一般还有其弟路易吉，偶尔或是其它马力欧角色。玩家操控角色在每关的平台上奔跑、跳跃，并从上方踩敌人头部。游戏剧情不复杂，典型的是马力欧营救被反派酷霸王绑架的桃花公主。系列首作是 1985 年红白机游戏《超级马力欧兄弟》，它确立了系列的基本游戏概念和元素。游戏中有各种道具，可以提升马力欧的能力，比如拥有投射火球的能力，或是变大缩小。

本次让强化学习 Agent 玩的游戏是最经典的，在红白机上的《**超级马力欧兄弟**》.

#### 强化学习基本概念

##### 基本介绍

强化学习(*Reinforcement Learning*, **RL**)讨论的问题是智能体(*agent*)怎么在一个复杂、不确定的环境(*environment*)里面去最大化它能获得的奖励. 示意图由两部分组成：智能体和环境. 在强化学习过程中, 智能体跟环境一直进行交互, 智能体在环境里面获取到状态(*state*), 并利用这个状态输出一个动作(*action*), 这个动作也被称为决策(*decision*). 然后这个决策会在环境中执行, 环境会根据智能体所采取的决策, 输出下一个状态以及当前的这个决策得到的奖励. 智能体的目的就是为了尽可能多地从环境中获取奖励. 

![img](E:\桌面\RL.assets\1.1.png)

近年来, 由于机器算力的提升和神经网络、深度学习的发展, 强化学习可以和深度学习结合到一起, 就形成了深度强化学习(*deep reinforcement learning*), 因此, 深度强化学习 = 深度学习 + 强化学习. 通过深度强化学习, 智能体与环境交互的过程就可以改进为一个端到端(*end-to-end*)的训练过程, 此时, 我们无需设计特征, 直接输入状态, 并定义神经网络的基本架构, 就可以输出 Action. 同时, 我们可以使用一个神经网络来拟合价值函数(*value function*), Q 函数或者策略网络, 省去特征工程(*feature engineering*)的过程.

##### 序列决策过程, Sequential Decision Making

强化学习研究的是智能体与环境交互的问题, 智能体会对环境输出动作, 环境取得动作之后进行下一步, 并把下一步的观测(observation)以及这个动作带来的奖励(reward)返还给 智能体. 智能体的目的就是从这些观测中学习到能够最大化奖励的策略. 

奖励是由环境给智能体的一个标量反馈信号, 这种信号可以显示智能体在某一步采取某个策略的表现如何. 强化学习的目的就是最大化智能体可以获得的奖励, 因此, 智能体在环境中存在的目的就是最大化期望的累计奖励（*expected cumulative reward*）. 

绝大多数的强化学习环境中, 智能体的奖励是被延迟了的了——我们选取的某一个动作, 可能需要很久之后才能知道这一步到底产生了什么影响. 强化学习中一个重要的课程就是近期奖励和远期奖励的权衡（*trade-off*）, 研究怎样让智能体获得更多的远期奖励.

在与环境交互的过程中, 智能体会获得很多观测, 针对每一个观测, 智能体会采取一个动作, 并得到一个奖励, 因此历史可以当做一个观测、动作、奖励的序列：
$$
H_t = o_1,a_1,r_1,\dots,o_t,a_t,r_t
$$
因此, 整个游戏的状态可以当做是关于这个历史的函数：
$$
s_t= f(H_t)
$$
需要注意的是状态和观测的关系：状态 $s$ 是对世界的完整描述, 不会隐藏世界的信息. 观测 $o$ 是对状态的部分描述, 可能会遗漏一些信息. 很多时候, 状态和观测是并不对等的. 当智能体只能够看到一部分的信息的时候, 我们就称这个环境是部分可观测(*partially observed*)的, 此时, 强化学习被建模为部分可观测的马尔科夫决策过程(*partially observable Markov decision process, POMDP*). 这个过程可以用一个七元组描述：
$$
(S,A,T,R,\Omega,O,\gamma)
$$
其中 $S$ 表示状态空间, 为隐变量, $A$ 为动作空间, $T(s'|s,a)$ 为状态转移概率, $R$ 为奖励函数, $\Omega(o|s,a)$ 为观测概率, O 为观测空间, $\gamma$ 为折扣系数. 

##### 动作空间, Action Space

不同的环境允许不同种类的动作. 在给定的环境中, 有效动作的集合经常被称为动作空间(*action space*). 像 Atari 和 Go 这样的环境有离散动作空间(*discrete action spaces*), 在这个动作空间里, agent 的动作数量是有限的. 在其他环境, 比如在物理世界中控制一个 agent, 在这个环境中就有连续动作空间(*continuous action spaces*) . 在连续空间中, 动作是实值的向量. 

##### 强化学习智能体的组成部分和类型

对于一个强化学习 agent, 它可能有一个或多个如下的组成成分：

- 策略函数(*policy function*), agent 会用这个函数来选取下一步的动作. 
- 价值函数(*value function*), 我们用价值函数来对当前状态进行估价, 它就是说你进入现在这个状态, 可以对你后面的收益带来多大的影响. 当这个价值函数大的时候, 说明你进入这个状态越有利. 
- 模型(*model*), 模型表示了 agent 对这个环境的状态进行了理解, 它决定了这个世界是如何进行的. 

其中, Policy 是 agent 的行为模型, 它决定了这个 agent 的行为, 它其实是一个函数, 把输入的状态变成行为. 强化学习一般使用随机性策略(*stochastic policy*), 其实也就是 $\pi$ 函数： $\pi(a | s)=P\left[A_{t}=a | S_{t}=s\right]$ . 当你输入一个状态 $s$ 的时候, 输出是一个概率. 这个概率是智能体所有动作的概率, 对这个概率分布进行采样, 就能得到智能体即将采取的动作.

价值函数是对未来奖励的预测, 我们用它来评估当前状态的好坏. 价值函数由一个期望定义：
$$
v_{\pi}(s) \doteq \mathbb{E}{\pi}\left[G{t} \mid S_{t}=s\right]=\mathbb{E}{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s\right], \text {for all } s \in \mathcal{S}
$$
其中, $\gamma$ 是一个折扣因子(*discount factor*). $\gamma$ 越大 agent 往前考虑的步数越多，但训练难度也越高；$\gamma$ 越小agent越注重眼前利益，训练难度也越小。我们都希望 agent 能“深谋远虑”，但过高的折扣因子容易导致算法收敛困难，因此作为强化学习的超参数，调整 $\gamma$ 的值是非常重要的. 总的来说，因此折扣因子的选择应当在算法能够收敛的前提下尽可能大. 

期望 $\mathbb E_\pi$ 的下标是 $\pi$ 函数, 它的值能够反映出我们在使用 $\pi$ 函数作为策略的时候, 到底能够得到多少奖励.

另一种价值函数被称为 $Q$ 函数, 定义如下：
$$
Q_{\pi}(s, a) \doteq \mathbb{E}{\pi}\left[G{t} \mid S_{t}=s, A_{t}=a\right]=\mathbb{E}{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s, A_{t}=a\right]
$$
$Q$ 函数包含了两个变量：状态和动作. 从 $Q$ 函数的定义也可以知道, 它就是我们在强化学习算法中需要学习的函数——因为当我们得到 $Q$ 函数之后, 进入某个状态需要采取的最优动作可以通过 $Q$ 函数得到.

第三个组成部分是模型, 它决定了下一步的状态. 下一步的状态取决于当前的状态以及当前采取的动作. 它由状态转移概率和奖励函数两个部分组成，状态转移概率也就是：
$$
p^a_{ss'}=p(s_{t+1}=s'|s_t=s, a_t=a)
$$
奖励函数表示我们在当前状态采取了某个动作，可以得到多少奖励：
$$
\mathcal{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_{t}=a\right]
$$
当有了策略、价值函数、模型三个部分之后，就形成了一个马尔科夫决策过程，这也是强化学习建模中常用的思路. 

##### 强化学习 Agent 的类型

根据智能体学习事物的不同, 可以将强化学习智能体分为基于价值的智能体(*value-based agent*) 以及基于策略的智能体(*policy-based agent*). 基于价值的智能体通过显式学习价值函数, 隐式地学习策略. 而基于策略的智能体直接学习策略, 给它一个状态, 它就会输出对应动作的概率. 将两种学习方式结合就有了演员-评论员智能体(*actor-critic agent*), 这类智能体通过学习策略函数、价值函数以及它们之间的交互得到最佳的动作.

我们也可以通过智能体是否学习环境模型将智能体分为有模型 (*model-based*) 以及免模型 (*model-free*) 智能体. 由于智能体在实际应用中较难估计状态转移函数和奖励函数, 甚至环境中的状态都是未知的, 因此需要采用免模型强化学习, 无需对真实环境建模. 免模型强化学习需要大量的采样来估计状态、动作以及价值函数, 从而优化策略. 大部分深度强化学习方法都采用了免模型强化学习.

##### 探索与利用

强化学习的目标是最大化 reward，我们不妨考虑较为简单的情形，也就是最大化单步奖励. 最大化单步奖励需要考虑两个方面：一是需要知道每个 action 带来的 reward，二是执行 reward 最大的动作. 一般来说，一个 action 的 reward 来自于一个概率分布，和环境交互的过程就是采样的过程，由此可知，仅仅通过一次尝试是无法确切地得到期望奖励的. 单步强化学习任务对应的于一个理论模型——K-臂赌博机(*K-armed bandit*). 

在强化学习中，探索(*exploration*)和利用(*exploitation*)是两个非常核心的问题. 探索是让我们尝试去探索环境，通过尝试不同的动作得到最佳的策略; 利用则是我们不去尝试新的动作，直接用当前已知的，能够带来最大奖励的策略. 对于探索和利用的关系，有以下值得注意的几个点：

1. 充分的探索才能带来有效的利用，从而使强化学习走在正确的道路上。对于那些难度特别高的任务，改进探索策略是性价比最高的手段;
2. 充分的利用才能探索到更好的状态，agent 往往需要掌握基本技能，才能解锁更高级的技能。就好像小孩先要学会站起来，才能学会走，然后才能学会跑。这种从易到难、循序渐进的思想在RL中也很受用;
3. 过量的探索阻碍及时的利用。如果随机探索噪声强度过高，已经学到的知识会被噪声淹没，而无法指导 agent解锁更好的状态，导致强化学习模型的性能停滞不前；
4. 机械的利用误导探索的方向。如果刚刚学到一点知识就无条件利用，agent 有可能被带偏，从而陷入局部最优，在错误道路上越走越远，在训练早期就扼杀了最好的可能性.

总而言之，强化学习是一个探索和利用的平衡游戏，前者使 agent 充分遍历环境中的各种可能性，从而有机会找到最优解；后者利用学到的经验指导 agent 做出更合理的选择。

#### DQN

##### DQN 基本介绍

传统的强化学习算法会使用表格的形式存储状态值函数 $V⁡(s)$ 或状态动作值函数 Q$⁡(s,a)$，但是这样的方法存在很大的局限性。例如：现实中的强化学习任务所面临的状态空间往往是连续的，存在无穷多个状态，在这种情况下，就不能再使用表格对值函数进行存储。值函数近似利用函数直接拟合状态值函数或状态动作值函数，减少了对存储空间的要求，有效地解决了这个问题。

为了在连续的状态空间中计算价值函数 $Q^{\pi}(s, a)$ ，我们可以用一个函数 $Q_{\phi}(\boldsymbol{s}, \boldsymbol{a})$ 来表示近似计算，称为价值函数近似*(Value Function Approximation*)。 
$$
Q_{\phi}(\boldsymbol{s}, \boldsymbol{a}) \approx Q^{\pi}(s, a)
$$
其中, $\boldsymbol{s}, \boldsymbol{a}$ 分别是状态 $s$ 和动作 $a$ 的向量表示，函数 $Q_\phi(s,a)$ 通常是一个参数为 $\phi$ 的函数，比如神经网络，输出为一个实数，称为 Q 网络(*Q-network*). 

深度 Q 网络（Deep Q-Network，DQN）算法的核心是维护 Q 函数并使用其进行决策。$Q_\pi(s, a)$ 为在该策略 $\pi$ 下的动作价值函数，每次到达一个状态 $s_t$ 之后，遍历整个动作空间，使用让 $Q_\pi⁡(s,a)$ 最大的动作作为策略：
$$
a_{t}=\underset{a}{\arg \max } ~Q^{\pi}\left(s_{t}, a\right)
$$
DQN 采用贝尔曼方程来迭代更新 $Q^{\pi}\left(s_{t}, a_{t}\right)$ ： 
$$
Q^{\pi}\left(s_{t}, a_{t}\right) \leftarrow Q^{\pi}\left(s_{t}, a_{t}\right)+\alpha\left(r_{t}+\gamma \max *{a} Q^{\pi}\left(s*{t+1}, a\right)-Q^{\pi}\left(s_{t}, a_{t}\right)\right)
$$
通常在简单任务上，使用全连接神经网络（*fully connected neural network*）来拟合 $Q_\pi$，但是在较为复杂的任务上（如玩 Atari Game），会使用卷积神经网络来拟合从图像到价值函数的映射。由于 DQN 的这种表达形式只能处理有限个动作值，因此其通常用于处理离散动作空间的任务。

##### Target Network

在学习 Q-function 的时候引入时序差分的思想，就可以推导出 Target Network 的想法了. 根据 Q 函数，我们可以知道：
$$
Q_{\pi}(s_t, a_t) =r_t + Q_\pi(s_{t+1}, \pi(s_t+1))
$$
因此，我们希望 $Q$ 函数输入 $s_t, a_t$ 得到的值和输入 $s_{t+1}, \pi(s_{t+1})$ 得到的值之间相差 $r_t$，这与时序差分的概念是一致的. 但是，如果假设我们把 $Q_{\pi} (s_t, a_t)$ 当作输出，把右端式子当作目标的话，此时目标是不断变动的，因此不太好训练.

Target Network 做的事情就是固定住其中一个 $Q$ 网络，并将其作为 Target，让另一个网络拟合它. 比如，在训练的时候，只更新左边网络的参数，而右边的参数固定住，这个时候就变成了一个回归问题. 在实现的时候，我们会把上式左边的 $Q$ 网络更新很多次，再用更新后的 $Q$ 网络替换目标网络.

##### Epsilon Greedy

强化学习的一大核心就在于解决探索-利用窘境(*exploration-exploitation dilemma*).

在 DQN 中，经常使用使用 $\epsilon -\text{ Greedy}$ 策略进行 action，也就是在选择策略的时候遵循以下原则：
$$
\begin{equation}
a =\left \{
\begin{aligned}
\arg \max_a Q(s, a) && \text{with propability }  1 - \epsilon
\\
random && \text{otherwise}
  
\end{aligned}
\right .
\end{equation}
$$
其中，$a$ 代表的当前选择的 action，$\epsilon$ 一般从 $1$ 开始逐渐递减. 这样的策略选择是有道理的：一开始的时候，由于需要充分探索环境，因此大部分的时候都是进行较为随机的策略，而当 Agent 充分探索环境之后，积累了较多经验，$\epsilon$ 逐渐收敛至较小的值，也就是选择当前模型觉得最好的 action.

##### Experience Replay

Experience Replay 会构建一个 Replay Buffer，Replay Buffer 又被称为 Replay Memory。Replay Buffer 是说现在会有某一个策略 $\pi$ 去跟环境交互，它会去收集数据。我们会把所有的数据放到一个 buffer 里面，buffer 里面保存着和当前环境相关的数据，也被称为 memory。比如说 buffer 是 5 万，这样它里面可以存 5 万条数据，每一条数据表示我们之前在某一个状态 $s_t$，采取某一个动作 $a_t$，得到了奖励 $r_t$。然后跳到状态 $s_{t+1}$. Replay Buffer 中存放着非常多不同策略的经验. 通过迭代训练 Q 函数的形式能够进行数据的更新——也就是从 buffer 中随机采样一个 batch，并根据这些经验更新 Q 函数.

##### Double DQN

在 DQN 的实现中，最经常使用到的 Tricks 就是 Double DQN 了. 强化学习天生过拟合，因此预测出来的 $Q$ 值往往会被高估，通过 Double DQN 的方式可以有效缓解这样的状态，让 Q Network 预测出的 value 接近真实值. 在实现的时候，有两个 $Q$ Network，其中一个是会更新参数的 $Q$ Network，而另一个是目标 $Q$ Network. 在 Double DQN 的实现中，我们用会更新参数的 $Q$ 选择 Action, 而用另一个网络来计算值.

#### 代码实现

##### 任务目标

通过 Reinforcement Learning 训练 Agent 玩超级玛丽游戏，并取得尽可能高的 Return.

##### 问题定义

为了简化问题，在 Super Mario Bros. 的环境中，Agent 的动作空间如下：
$$
\set{walk, jump}
$$
其中 $walk$ 代表让 Mario 向右走，而 $jump$ 代表让 Mario 向右跳.

Agent 的状态空间被定义为 shape 为 $(4,84,84)$ 的 Tensor，每一个 Tensor 代表着 Mario Game 的 4 个 Frame，$(84,84)$ 则是每一个 Frame 的大小. 初始的每一帧经过灰度化，帧叠加，最终会得到一个上述 shape 的 Tensor. Reward 则是 Mario 在每个 step 拿到的分数.

##### 实验环境

**软件**: Windows 11, Python 3.9.9 with torch-gpu

**GPU**: Nvidia GTX 1050

##### Mario 环境准备

OpenAI 的 gym 已经有了非常多封装好的环境可供选择，对于经典的 Mario Game 当然也不例外，`gym_super_mario_bros` 就是一个已经封装好的强化学习环境，它能够让强化学习环境

在具体实现的时候，由于每一帧之间的变化并不是很大，从节省资源的角度来说，我们可以对每 4 帧进行一次 Sampling，将其灰度化，并将每 4 个 Frame 打包，转化为 Tensor. 代码实现方面没有什么太 tricky 的东西，具体的代码可以看 `Wrappers.py` 中的实现.

```Python
# 0. 导入环境
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0") # 导入环境
env = JoypadSpace(env, [["right"], ["right", "A"]]) # 限制 Action 在向右走和向右跳之间
env = wrappers(env)  # 套上 wrapper, 让数据相对容易处理些
```

##### MarioNet

深度学习的引入能够将每一个 Frame 转化为一个 Feature Vector 输入模型，从而实现端到端训练 model. MarioNet 做的也是这样一件事，它将上述环境中经过预处理的 Tensor 转化为一个 Feature Vector.

DDQN 的其中一个网络定义如下（实际上，两个网络的架构是一致的，只是一个被固定住参数，而另一个在学习时不断更新参数）：

```Python
from torch import nn
# Mini-CNN
# input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
nn.Sequential(
    nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(3136, 512),
    nn.ReLU(),
    nn.Linear(512, output_dim),
)
```

##### Agent - Mario

Agent 是强化学习中重要的组成部分.  对于 Agent 来说需要实现的核心操作就是和环境交互，也就是让模型根据当前的 `state` 选择 `action`.  这里使用上面提到的 $\text{Epsilon Greedy}$ 策略和环境进行交互.

```Python
 def act(state):
    # Epsilon Greedy
    if np.random.rand() < exploration_rate:
        # 探索, 从 action space 中随机取样
        action_idx = np.random.randint(action_dim)
    else:
        state = state_preprocess(state)
        action_values = net(state, model="online")
        # 利用, 取出当前 model 觉得最好的 action
        action_idx = torch.argmax(action_values, axis=1).item()
        # 降低当前的 exploration rate
        exploration_rate = max(exp_rate_min, exp_rate * exp_rate_decay)
        curr_step += 1  # 增加当前的 step

        return action_idx
```

##### Learning

通过上述 DQN 理论的介绍，我们在学习步骤需要做的就是让 $Q_{\text{online}}$ 的值逼近 $Q_{\text{target}}$. 核心代码如下：

```Python
def update_Q_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()  # 通过反向传播的方式更新 update 的参数
    self.optimizer.step()
    return loss.item()

def sync_Q_target(self):
    self.net.target.load_state_dict(self.net.online.state_dict())
```

训练的结果如下：



##### Play The Game

在强化学习中，predict 的过程也就是利用 model 玩游戏的过程，代码和 Agent 在训练的时候进行交互的核心代码几乎是一致的，只是有可能需要进行一些预处理，具体的处理方式可以看源代码.  

```Python
# 这里只展示了一个 Episode 的
state = env.reset() # 开始游戏
while True:
    state = state.transform() # 将当前的 state 转化为 model 能够处理的样子
    action, _ = model.predict(state) # 使用上述模型在当前的 state 下输出一个 action
    state, reward, done, info = env.step(action) # 和 Enviornment 交互, 得到下一个 action
    if done or info['flag_get']:  # 当 game_over 或者拿到最终的 flag 的时候, 游戏结束
		   break
    env.render() # 渲染当前帧
```

#### 待改进的部分

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
- 更换算法，尝试使用 PPO 或 A3C 进行 Training

##### 可视化程度不足

在本次 Training 中，只使用了 matplotlib 进行可视化，未来可能考虑用 TensorBoard 实现可交互的可视化.

##### 没有写在 CPU 上 Training 的代码

为了简化思考以及 coding 的难度，本项目并没有写在 CPU 上 Train 的代码（因为要加很多判断 `cuda` 是否 `available` 的分支，比较麻烦），后续会将其补上.

#### 总结

这次包括决定开始做强化学习的项目，到从 0 开始寻找、摸索各种强化学习的学习方式，再到搭建坦克大战 Simulator 的失败（一开始想从 battle-city.js 中构建 simulator，但是因为自己前端知识的不牢固，最终放弃了这条路；之后找到了 battle-city 的 `nes` 文件，用 Gym Retro 做 Game Integration，但是因为只能找到几个 variables，尝试强行进行训练，但是最后也作罢了） ，从而转向 Mario 的全过程，让我深深地感慨自己在编写代码这件事情上还有非常大的提升空间——一旦写一些不太 naive 的，需要自己深刻思考的东西的时候，并没有拆解任务的意识，而是一味地畏惧困难直到放弃或者选择更简单的东西，这也是我学习东西的劣根性所在. 我更是深知自己和合格的“炼丹师”有着非常远的距离.

但是从 0 开始学习强化学习的过程至少给了我一些收获：对于强化学习的概念有了最基本的了解、能够通过 Tutorial 自己写出代码训练一个强化学习的 model（尽管因为各种原因的限制没能达到很完美的程度，甚至可以说是比较失败的）、对于 DQN 的认知上了一个层次，对于其他强化学习的算法，比如最基础的 Q-learning, 再到 PPO, A3C, DDPG 等算法也有一定的了解. 对于强化学习中的一些关键问题，也有了皮毛程度的了解. 在寻找强化学习相关资料的时候，更是能够感觉到自己的信息检索能力以及阅读英文文章的能力正在一点点地提升. 使用 GitHub 和相关工具的时候也变得稍微得心应手了一些.

从宏观的角度来说，强化学习的一些核心思想也可以运用到实际的学习中——解决稀缺奖励、奖励延迟问题就需要进行 reward shaping，在实际的学习中我们也常常因为延迟奖励而直接放弃，如果能够有较为及时的反馈，学习过程也就更容易坚持下来，训练 Agent 的过程感觉甚至有些像是在养一个到处乱撞的孩子.

总而言之
