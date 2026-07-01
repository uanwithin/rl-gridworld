# rl-gridworld 项目结构说明

这份文档从高层到细节梳理当前仓库的组织方式，并说明这个结构是否适合继续扩展。

## 总体评价

当前项目组织是合理的，尤其适合作为强化学习入门和算法对比仓库。

它把几个容易混在一起的职责拆开了：

- `env` 只描述世界规则，也就是 GridWorld 这个 MDP。
- `core` 放公共数据类型，比如策略和值函数。
- `agents` 放算法实现，比如 Value Iteration。
- `gui` 放终端和图形可视化。
- `cli.py` 把环境、算法和可视化串起来。
- `tests` 验证环境和算法的基础行为。

这种分层的好处是：后续新增算法时，通常只需要在 `agents` 下加文件，并在 `cli.py` 注册；只要算法输出 `Policy`，原有可视化就可以复用。

我建议暂时不要大规模重构核心代码。这个仓库目前是教学型项目，清晰比抽象更重要。现在最值得做的是保持文档和实际代码一致，并在新增算法时继续沿用当前接口。

## 高层数据流

项目运行时的主流程是：

```text
cli.main()
  ↓
build_gridworld()
  ↓
GridWorld(P, R, next_states)
  ↓
Agent.solve(env)
  ↓
Policy / ValueFunction / ActionValueFunction
  ↓
render_ascii() / plot_value() / plot_policy()
```

换句话说：

```text
环境定义问题
算法求解问题
公共类型承载结果
可视化展示结果
CLI 负责调度
```

## 目录结构

当前主要代码结构如下：

```text
src/rl_gridworld/
├── env/
│   └── grid_world.py
├── core/
│   └── types.py
├── agents/
│   ├── base.py
│   ├── dp/
│   │   └── value_iteration.py
│   ├── mc/
│   └── td/
├── gui/
│   ├── ascii.py
│   └── plots.py
└── cli.py

tests/
├── test_grid_world.py
└── test_value_iteration.py
```

## env: 环境层

文件：`src/rl_gridworld/env/grid_world.py`

这一层定义 GridWorld，也就是整个项目要解决的问题。

当前环境是一个 5x5 的确定性 MDP：

- 状态数：`25`
- 动作数：`5`
- 动作：上、下、左、右、原地不动
- 目标格：`TARGET = (3, 2)`
- 禁区：`FORBIDDEN = [(1, 1), (1, 2), (2, 2), (3, 1), (4, 1)]`

核心数据有三张表：

```text
P[s, a, s2]       从状态 s 执行动作 a 转移到 s2 的概率
R[s, a]           从状态 s 执行动作 a 得到的即时奖励
next_states[s,a]  确定性环境下的下一个状态
```

奖励规则是：

```text
撞墙: -1
进入禁区: -1
进入目标: +1
进入普通格子: 0
```

注意：当前 `TARGET` 不是终止状态。也就是说，到达目标后并不会自动结束 episode。如果在目标格选择原地不动，仍然会再次得到目标奖励。这个设定对 Value Iteration 的结果有直接影响，目标状态通常会拥有最高价值。

## core: 公共类型层

文件：`src/rl_gridworld/core/types.py`

这一层定义算法和可视化共享的数据结构。

`Policy` 表示策略：

```text
Policy.probs.shape == (n_states, n_actions)
```

每一行是一个状态下各动作的概率。确定性策略就是每行只有一个动作概率为 `1.0`。

`ValueFunction` 表示状态价值：

```text
V(s)
```

对应数组形状是：

```text
(n_states,)
```

`ActionValueFunction` 表示动作价值：

```text
Q(s, a)
```

对应数组形状是：

```text
(n_states, n_actions)
```

它还提供 `greedy_policy()`，可以从 `Q` 直接取贪婪策略。

这一层的存在让所有算法有统一输出格式。可视化层只需要认识 `Policy` 和 `ValueFunction`，不需要知道算法内部是动态规划、蒙特卡洛还是时序差分。

## agents: 算法层

文件：`src/rl_gridworld/agents/base.py`

所有算法都继承 `Agent`，并实现：

```python
def solve(self, env: GridWorld) -> Policy:
    ...
```

这个接口表示：

```text
输入一个环境
输出一个策略
```

当前已经实现的算法是：

```text
src/rl_gridworld/agents/dp/value_iteration.py
```

### Value Iteration

Value Iteration 是有模型动态规划算法，直接使用环境中的 `P` 和 `R`。

核心更新公式是：

```text
Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')
V(s)   = max_a Q(s,a)
```

代码里的流程是：

1. 初始化 `V` 为全 0。
2. 根据当前 `V` 计算所有 `Q(s,a)`。
3. 对每个状态取最大动作价值，得到新的 `V`。
4. 比较新旧 `V` 的最大差值 `delta`。
5. 如果 `delta < theta`，认为收敛。
6. 收敛后重新计算 `Q`。
7. 保存 `self.V`、`self.Q` 和 `self.n_iters`。
8. 返回 `self.Q.greedy_policy()`。

这种写法很适合教学，因为它几乎直接对应 Bellman 最优方程。

## gui: 可视化层

文件：

- `src/rl_gridworld/gui/ascii.py`
- `src/rl_gridworld/gui/plots.py`

`ascii.py` 提供终端渲染：

- 普通格子显示为空白。
- 禁区显示为 `███`。
- 目标显示为旗帜。
- 如果传入 agent 状态，则显示 agent。

`plots.py` 提供 matplotlib 可视化：

- `plot_value(value)` 把 `V(s)` 画成 5x5 热力图。
- `plot_policy(policy)` 把策略画成箭头图。
- `plot_returns(returns)` 预留给未来 model-free 算法画学习曲线。

当前 `plot_policy` 展示的是每个状态的贪婪动作。如果未来要展示随机策略，可以扩展成在一个格子里画多个带权重的箭头。

## cli.py: 命令行入口

文件：`src/rl_gridworld/cli.py`

CLI 负责把所有模块串起来：

1. 解析命令行参数。
2. 构造默认 GridWorld。
3. 打印环境布局。
4. 根据算法名创建 agent。
5. 调用 `agent.solve(env)`。
6. 输出收敛信息。
7. 可选地弹出价值图和策略图。

算法注册表是：

```python
ALGORITHMS = {
    "value_iteration": ValueIteration,
}
```

新增算法时，通常只需要把新算法类加进这个字典。

## tests: 测试层

当前测试覆盖两部分。

`tests/test_grid_world.py` 验证环境：

- `P` 和 `R` 的形状正确。
- 每个状态动作对的转移概率和为 1。
- 撞墙会留在原状态并得到边界惩罚。
- 状态编号和网格坐标可以互相转换。
- 进入目标格会得到目标奖励。

`tests/test_value_iteration.py` 验证算法：

- Value Iteration 能运行并收敛若干轮。
- 输出策略形状正确。
- `agent.V` 会被填充。
- 目标状态价值达到全局最大。

## 当前结构是否需要继续修改

短期内不需要大改。当前结构已经足够清晰，适合继续往下实现算法。

我建议保留以下约定：

1. `env` 不依赖算法和可视化。
2. `core` 不依赖具体算法。
3. `agents` 可以依赖 `env` 和 `core`，但不要依赖 `gui` 或 `cli`。
4. `gui` 尽量只依赖 `core` 的结果类型，以及必要的网格布局信息。
5. `cli` 是装配层，可以依赖所有模块。

后续如果项目变复杂，可以考虑这些改动：

- 给 `GridWorld` 增加 `done` 或 terminal-state 机制。
- 增加起点、episode reset、随机转移等接口，支持 Monte Carlo 和 TD 更自然地采样。
- 给 `Agent` 增加可选的训练统计信息，比如 `returns`、`losses`、`steps`。
- 把环境布局从常量改成配置对象，支持不同地图。
- 给 CLI 增加 `--theta`、`--max-iters`、`--episodes` 等算法参数。
- 当算法数量变多时，把算法注册从硬编码字典改成更系统的注册机制。

## 新增算法建议

新增一个算法时，建议遵循这个路径：

1. 在对应目录下创建算法文件，比如 `agents/dp/policy_iteration.py`。
2. 继承 `Agent`。
3. 实现 `solve(env) -> Policy`。
4. 如果算法会产生状态价值或动作价值，填充 `self.V` 或 `self.Q`。
5. 在 `cli.py` 的 `ALGORITHMS` 中注册。
6. 增加一个最小测试，确保算法能输出合法策略。

例如，下一步最自然的算法是 Policy Iteration。它和 Value Iteration 一样都是有模型 DP 算法，可以直接复用当前 `GridWorld.P` 和 `GridWorld.R`，对项目结构的冲击最小。

## 一句话总结

这个仓库现在的结构是一个干净的强化学习教学骨架：`env` 提供 MDP，`agents` 求解策略，`core` 统一结果类型，`gui` 负责展示，`cli` 负责运行入口，`tests` 保证基础行为。当前更适合稳步添加算法，而不是提前做复杂抽象。
