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
```

## 目录结构

```
src/rl_gridworld/
├── env/        环境: GridWorld (确定性 MDP, P / R / next_states)
├── core/       RL 公共抽象: Policy / ValueFunction / ActionValueFunction
├── agents/     算法, 按家族分组
│   ├── base.py   Agent 抽象基类: solve(env) -> Policy
│   ├── dp/       有模型 (动态规划): value_iteration ...
│   ├── mc/       蒙特卡洛
│   └── td/       时序差分: sarsa / q_learning ...
├── gui/        可视化: ascii (终端) / plots (matplotlib) / qt_view (待实现)
└── cli.py      入口, 把环境 + 算法 + 可视化粘起来
tests/          单元测试
```

设计约定：依赖方向单向为 `gui → agents → core → env`。所有算法的输出都对齐到
`core.types.Policy`，因此可视化层用同一套代码即可渲染任意算法的结果，方便横向对比。

## 已实现算法

- [x] Value Iteration (`agents/dp`)
- [ ] Policy Iteration
- [ ] Monte Carlo
- [ ] Sarsa / Q-learning
- [ ] 价值函数近似 / 策略梯度

## 新增一个算法

1. 在对应家族目录 (如 `agents/td/`) 新建文件，继承 `Agent` 并实现 `solve`。
2. 在 `cli.py` 的 `ALGORITHMS` 注册表里登记。
3. 输出 `Policy`（如有需要也填充 `self.V` / `self.Q`）即可直接被可视化。

## 开发

```bash
uv run pytest        # 测试
uv run ruff check .  # lint
uv run ruff format . # 格式化
```
