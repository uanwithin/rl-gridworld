# rl-gridworld

以 Grid-World 为载体，循序渐进地实现并可视化各种强化学习算法。适合入门学习。

## 安装

```bash
uv sync
```

## 运行

```bash
# 运行默认算法 (value iteration) 并弹出价值/策略图
uv run rl-gridworld

# 指定算法与折扣因子，只在终端输出不弹窗
uv run rl-gridworld value_iteration --gamma 0.9 --no-plot

# 评估均匀随机策略（迭代法）；去掉 --no-plot 可显示价值和策略图
uv run rl-gridworld policy_evaluation --gamma 0.9 --no-plot

# 策略迭代：交替评估和改进策略
uv run rl-gridworld policy_iteration --gamma 0.9 --no-plot
```

## 按键单步绘图

```bash
uv run rl-gridworld --step
uv run rl-gridworld policy_iteration --step
uv run rl-gridworld policy_evaluation --step --quiet
```

窗口初始显示第 0 轮（价值全为 0，策略均匀随机）。点击图窗使其获得键盘焦点后：

- 空格、回车、右方向键或 `Next` 按钮：实际执行下一轮迭代。
- `R` 或 `Reset` 按钮：回到初始状态。
- 收敛或达到迭代上限后停止更新，可重置后重新观察。

三个面板分别显示当前价值、本轮价值差值、当前策略；策略变化的格子用红框标记。
策略迭代的一轮包含“准确评估 + 策略改进”，因此收敛前显示的价值是改进前策略的评估值。
终端仍显示每轮具体变化，使用 `--quiet` 可隐藏明细。`--step` 不能与 `--no-plot` 同用。
图窗使用英文标签，以避免系统缺少中文字体时出现方框。

## 查看每轮更新

默认在终端显示每轮更新：状态编号、坐标（从 0 开始）、价值旧值 → 新值、
差值和最大变化。策略迭代分别显示评估与改进阶段，并列出动作概率变化。
未变化的状态省略，整轮无变化时明确标注。图形窗口仍展示最终结果。

```bash
uv run rl-gridworld value_iteration --no-plot
uv run rl-gridworld policy_iteration --no-plot
# 仅查看最终结果
uv run rl-gridworld --quiet --no-plot
```

Python 调用可传入 `on_update` 回调：

```python
updates = []
solution = ValueIteration().solve(env, on_update=updates.append)
```

每条更新含 `iteration`、`phase`、`before`、`after`，数组为独立快照。
`evaluate_policy_iterative` 和 `PolicyIteration.solve` 也支持此参数；
截断策略迭代逐步报告内部评估更新。

## 目录结构

```
src/rl_gridworld/
├── env/        环境: GridWorld (确定性 MDP, 可自定义地图, 提供 P / R / next_states)
├── core/       RL 公共类型: Policy / ValueFunction / ActionValueFunction / Solution
├── agents/     算法, 按家族分组
│   ├── base.py   Agent 抽象基类: solve(env) -> Solution
│   ├── dp/       有模型 (动态规划): policy_evaluation / value_iteration ...
│   ├── mc/       蒙特卡洛
│   └── td/       时序差分: sarsa / q_learning ...
├── gui/        可视化: ascii (终端) / plots (matplotlib)
└── cli.py      入口, 把环境 + 算法 + 可视化粘起来
tests/          单元测试
```

设计约定：`env` 定义环境模型，`core` 定义算法和可视化共享的数据类型，
`agents` 放算法，`gui` 只负责渲染这些公共类型，`cli` 负责把环境、算法和可视化粘起来。

算法分两类：

- **控制**：找出好的策略，如值迭代、策略迭代、蒙特卡洛、Sarsa、Q-learning。
  写成继承 `Agent` 的类，`solve(env) -> Solution`，在 `cli.py` 注册返回 `Solution` 的调用函数后就能运行和可视化。
- **预测**：评估一个给定的策略，如策略评估、TD 状态值估计。
  写成函数 `(env, policy, gamma, ...) -> ValueFunction`，供控制算法调用，也在测试和实验里当基准。

因此新增控制算法时，通常不需要改可视化代码。

更详细的结构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 已实现算法

- [x] 策略评估 (`agents/dp`，预测)
- [x] Value Iteration (`agents/dp`)
- [x] Policy Iteration
- [ ] 截断策略迭代
- [ ] Monte Carlo: MC Basic / Exploring Starts / ε-Greedy
- [ ] TD: 状态值估计 (预测) / Sarsa / n-step Sarsa / Q-learning
- [ ] 价值函数近似 / 策略梯度

## 新增一个算法

控制算法：

1. 在对应家族目录 (如 `agents/td/`) 新建文件，继承 `Agent` 并实现 `solve`。
2. `solve` 返回 `Solution`：至少填 `policy`，有的话再填 `V` / `Q` / `n_iters` / `returns`。
3. 在 `cli.py` 的 `ALGORITHMS` 注册表里登记一个构造函数，结果就能直接被可视化。

预测算法：在对应家族目录里写一个返回 `ValueFunction` 的函数。需要命令行实验时，
可在 `cli.py` 中包装成返回 `Solution` 的调用函数并注册；`policy_evaluation`
已按此方式接入，默认评估均匀随机策略。
测试里用 `evaluate_policy_exact` 的结果当标准答案。

## 自定义地图

```python
from rl_gridworld.env import GridWorld, Rewards

env = GridWorld(
    rows=4, cols=6, target=(3, 5), forbidden=((1, 1), (2, 3)),
    rewards=Rewards(forbidden=-10.0),
)
```

## 开发

```bash
uv run pytest        # 测试
uv run ruff check .  # lint
uv run ruff format . # 格式化
```
