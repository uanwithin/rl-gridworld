# rl-gridworld 项目结构说明

这份文档从高层到细节梳理仓库的组织方式，以及新增算法时应该遵循的约定。

## 分层

- `env` 只描述世界规则，也就是 GridWorld 这个 MDP。
- `core` 放公共数据类型：策略、值函数和求解结果 `Solution`。
- `agents` 放算法实现，分为控制算法（`Agent` 子类）和预测算法（函数），见下文。
- `gui` 放终端和图形可视化。
- `cli.py` 把环境、算法和可视化串起来。
- `tests` 验证环境、算法和可视化的基础行为。

依赖方向是单向的：

1. `env` 不依赖算法和可视化。
2. `core` 不依赖具体算法。
3. `agents` 依赖 `env` 和 `core`，不依赖 `gui` 或 `cli`。
4. `gui` 只依赖 `core` 的结果类型和 `GridWorld` 的地图布局。
5. `cli` 是装配层，可以依赖所有模块。

这样新增控制算法时，只需要在 `agents` 下加文件并在 `cli.py` 注册；只要算法返回 `Solution`，现有可视化就能直接复用。

## 高层数据流

```text
cli.main()
  ↓
build_gridworld()  →  GridWorld(P, R, next_states, 地图布局)
  ↓
Agent.solve(env)
  ↓
Solution(policy, V, Q, n_iters, returns)
  ↓
render_grid / render_values / render_policy  (终端)
plot_solution                                 (matplotlib)
```

## 目录结构

```text
src/rl_gridworld/
├── env/
│   └── grid_world.py
├── core/
│   └── types.py
├── agents/
│   ├── base.py
│   ├── dp/
│   │   ├── policy_evaluation.py
│   │   └── value_iteration.py
│   ├── mc/
│   └── td/
├── gui/
│   ├── ascii.py
│   └── plots.py
└── cli.py

tests/
├── test_grid_world.py
├── test_value_iteration.py
└── test_gui.py
```

## env: 环境层

文件：`src/rl_gridworld/env/grid_world.py`

`GridWorld` 由地图布局构造，构造时一次性推导出完整的 MDP 模型：

```python
GridWorld(rows=5, cols=5, target=(3, 2), forbidden=(...), rewards=Rewards())
```

默认地图（`build_gridworld()`）是一个 5x5 的确定性 MDP：

- 状态数：`25`，按行优先编号 `s = r * cols + c`，(0, 0) 是左上角
- 动作：`Action.UP / DOWN / LEFT / RIGHT / STAY`，位移定义在 `ACTION_DELTAS`
- 目标格：`TARGET = (3, 2)`
- 禁区：`FORBIDDEN = ((1, 1), (1, 2), (2, 2), (3, 1), (4, 1))`

核心数据有三张表：

```text
P[s, a, s2]       从状态 s 执行动作 a 转移到 s2 的概率
R[s, a]           从状态 s 执行动作 a 得到的即时奖励
next_states[s,a]  确定性环境下的下一个状态
```

`env.step(s, a) -> (s2, r)` 是查表的便捷写法，后续 model-free 算法可以用它采样。

奖励由 `Rewards` 配置，默认值是：

```text
撞墙: -1   (留在原地)
进入禁区: -1   (允许进入)
进入目标: +1
进入普通格子: 0
```

注意：`TARGET` 不是终止状态。在目标格选择原地不动，仍然会再次得到目标奖励，所以目标格的最优价值是 `1 / (1 - gamma)`。另外禁区惩罚只有 -1 时，最优策略会愿意穿过禁区抄近路；把 `Rewards(forbidden=-10)` 之类调大可以观察策略的变化。

## core: 公共类型层

文件：`src/rl_gridworld/core/types.py`

- `Policy`：`probs.shape == (n_states, n_actions)`，每行是一个状态下各动作的概率。
- `ValueFunction`：状态价值 `V(s)`，形状 `(n_states,)`。
- `ActionValueFunction`：动作价值 `Q(s, a)`，形状 `(n_states, n_actions)`，提供 `greedy_policy()`。
- `Solution`：一次求解的结果。

```python
@dataclass
class Solution:
    policy: Policy
    V: ValueFunction | None = None
    Q: ActionValueFunction | None = None
    n_iters: int = 0              # DP: 迭代轮数; model-free: episode 数
    returns: list[float] = ...    # 每个 episode 的回报, 用来画学习曲线
```

不同算法填充不同字段：DP 算法通常有 `V`、`Q`、`n_iters`；Q-learning 之类可能只有 `Q` 和 `returns`。可视化层按需取用，不关心算法内部。

## agents: 算法层

算法按「输入什么、输出什么」分成两类：

| 类型 | 问题 | 写法 | 例子 |
|---|---|---|---|
| 控制 | 给一个环境，找出好的策略 | 继承 `Agent`，`solve(env) -> Solution` | 值迭代、策略迭代、MC、Sarsa、Q-learning |
| 预测 | 给一个策略，求它的价值 | 函数 `(env, policy, gamma, ...) -> ValueFunction` | 策略评估、TD 状态值估计 |

控制算法是命令行里直接运行的对象。构造时只保存超参数（`gamma`、`theta` 等），结果全部通过 `Solution` 返回，同一个算法对象可以反复求解不同环境。

预测算法的输入多了一个策略、输出是价值而不是策略，套不进 `solve(env) -> Solution`，也没有需要在调用之间保存的状态，所以写成函数。它们的 `gamma` 必须由调用方显式传入，超参数的默认值只放在 `Agent` 和 CLI 上。

一段计算只有一个算法用，就写成该算法的私有方法或模块私有函数；多个算法共用，就写成模块级的公开函数。

### 完整路线

对应 `docs/` 里的学习笔记：

| 笔记 | 算法 | 类型 | 位置 | 和策略评估的关系 |
|---|---|---|---|---|
| 02 | 策略评估 | 预测 | `dp/policy_evaluation.py` | 本身 |
| 04 | 值迭代 | 控制 | `dp/value_iteration.py` | 共用 `q_from_v` |
| 04 | 策略迭代 / 截断策略迭代 | 控制 | `dp/policy_iteration.py` | 评估步调用 `evaluate_policy_iterative`，改进步用 `q_from_v` |
| 05 | MC Basic / Exploring Starts / ε-Greedy | 控制 | `mc/` | 测试里用解析解的 `q_π` 对照采样估计 |
| 07 | TD 状态值估计 | 预测 | `td/` | 和解析解对比估计误差 |
| 07 | Sarsa / n-step Sarsa / Q-learning | 控制 | `td/` | 结果可与值迭代对比 |

策略评估有策略迭代、测试、无模型算法的对照等多个使用方，所以是 `dp/` 里的公开函数，而不是策略迭代的成员方法。

### Policy Evaluation

文件：`src/rl_gridworld/agents/dp/policy_evaluation.py`

求解贝尔曼方程 `v_π = r_π + γ P_π v_π`：

- `evaluate_policy_exact(env, policy, gamma)`：解线性方程组。状态少时又快又准，是测试里的标准答案。
- `evaluate_policy_iterative(env, policy, gamma, theta, max_iters, v0)`：迭代 `v <- r_π + γ P_π v`。
- `q_from_v(env, v, gamma)`：向前看一步算 `q(s,a) = r(s,a) + γ Σ_s' p(s'|s,a) v(s')`。值迭代和策略迭代的改进步都用它。
- `_policy_reward` / `_policy_transition`：算 `r_π` 和 `P_π`，只供上面两个评估函数使用。

迭代解的 `v0` 和 `max_iters` 正好是「评估从哪里开始、跑几步」两个旋钮：跑到收敛是策略迭代；用上一轮的值热启动、只跑有限步是截断策略迭代。跑满 `max_iters` 时直接返回当前估计、不发警告。

### Value Iteration

文件：`src/rl_gridworld/agents/dp/value_iteration.py`

有模型动态规划算法，直接使用环境中的 `P` 和 `R`：

```text
Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')
V(s)   = max_a Q(s,a)
```

1. 初始化 `V` 为全 0。
2. 用 `q_from_v` 根据当前 `V` 计算所有 `Q(s,a)`，对每个状态取最大值得到新的 `V`。
3. 如果新旧 `V` 的最大差值 `delta < theta`，认为收敛；超过 `max_iters` 仍未收敛会发出警告。
4. 用最终的 `V` 计算 `Q`，返回 `Solution(policy=Q 的贪婪策略, V, Q, n_iters)`。

## gui: 可视化层

所有可视化函数都把 `env` 作为第一个参数，从中读取网格尺寸、目标和禁区，因此自定义地图也能直接画。

`ascii.py`（返回字符串，由调用方打印）：

- `render_grid(env, agent_state=None)`：地图布局，禁区显示为 `███`，目标显示为旗帜。
- `render_values(env, value)`：`V(s)` 数字表。
- `render_policy(env, policy)`：每格的贪婪动作，`↑ ↓ ← →` 和原地不动 `o`。

`plots.py`（matplotlib）：

- `plot_value(env, value)`：`V(s)` 热力图。
- `plot_policy(env, policy)`：箭头图，箭头长度与 `π(a|s)` 成正比，确定性策略和 ε-greedy 策略都能画。箭头方向直接取自 `ACTION_DELTAS`，和环境保持一致。
- `plot_returns(returns)`：学习曲线，给 model-free 算法用。
- `plot_solution(env, solution)`：把 `Solution` 里已有的内容并排画在一张图上。

## cli.py: 命令行入口

1. 解析命令行参数（`main(argv)` 接受参数列表，方便测试）。
2. 构造默认 GridWorld 并打印布局。
3. 根据算法名创建 agent，调用 `agent.solve(env)`。
4. 在终端打印迭代次数、`V(s)` 和策略。
5. 除非传入 `--no-plot`，弹出 matplotlib 图。

算法注册表把名字映射到「由命令行参数构造 agent」的函数，每个算法可以取自己需要的参数：

```python
ALGORITHMS = {
    "value_iteration": lambda args: ValueIteration(gamma=args.gamma),
}
```

## 新增算法建议

控制算法：

1. 在对应目录下创建算法文件，比如 `agents/dp/policy_iteration.py`。
2. 继承 `Agent`，实现 `solve(env) -> Solution`。
3. 在 `cli.py` 的 `ALGORITHMS` 中注册；如果需要新超参数（如 `--episodes`、`--epsilon`），同时在 `parse_args` 里加上。
4. 增加一个最小测试，确保算法能输出合法策略；可以和 Value Iteration 的结果对比。

预测算法：在对应目录里写一个返回 `ValueFunction` 的函数，不在 `cli.py` 注册；测试里和 `evaluate_policy_exact` 的结果对比。

策略迭代和截断策略迭代只差评估步数，可以是同一个 `PolicyIteration` 类加一个评估步数参数，在 `cli.py` 里注册成两个名字。

开始做 Monte Carlo / TD 时，可能需要给环境增加起点、episode 长度或终止状态等概念，建议到那时按算法的实际需要再加。
