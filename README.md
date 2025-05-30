# Portfolio Optimization Framework

这是一个多策略投资组合优化框架，提供了多种投资策略的实现和回测功能。

## 功能特点

- 多种投资策略实现
  - 均值方差策略 (Mean-Variance)
  - 风险平价策略 (Risk Parity)
  - 条件风险平价策略 (Conditional Risk Parity)
  - 动量策略 (Momentum)
    - 最大动量策略
    - 动量阈值策略
  - 分层策略 (Hierarchical)
    - 分层拉菲诺策略
    - 层级动量策略
  - 凯利公式策略 (Kelly Criterion)

- 完整的回测框架
  - 支持多策略并行回测
  - 灵活的再平衡周期设置
  - 详细的性能指标计算

- 丰富的可视化功能
  - 累计收益对比
  - 回撤分析
  - 滚动性能指标
  - 策略相关性热图

## 项目结构

详细的项目结构请参见 `project_structure.txt`

## 快速开始

1. 克隆项目到本地
2. 安装依赖包
3. 运行示例代码：

```python
python examples/multi_strategy_demo.py
```

## 示例代码

```python
from portfolio_optimization.data.random_generator import RandomDataGenerator
from portfolio_optimization.strategies.mean_variance import MeanVarianceStrategy
from portfolio_optimization.backtest.backtest_engine import BacktestEngine

# 生成模拟数据
data_generator = RandomDataGenerator(assets=['A', 'B', 'C'], seed=42)
prices, returns = data_generator.load_data(
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# 初始化策略
strategy = MeanVarianceStrategy(prices=prices, returns=returns)

# 运行回测
backtest_engine = BacktestEngine(prices=prices, returns=returns)
portfolio_values, weights_history = backtest_engine.run_backtest(
    strategy=strategy,
    start_date='2022-01-01',
    end_date='2024-12-31',
    rebalance_freq='M'
)
```

## 添加新策略

要添加新的策略，只需：

1. 在 `strategies/` 目录下创建新的策略类
2. 继承 `BaseStrategy` 类
3. 实现 `generate_weights` 方法

示例：
```python
from portfolio_optimization.strategies.base_strategy import BaseStrategy

class MyNewStrategy(BaseStrategy):
    def generate_weights(self, date: str, **kwargs) -> Dict[str, float]:
        # 实现权重生成逻辑
        return weights
```

## 注意事项

- 回测时请确保数据长度足够，建议预留足够的历史数据用于策略计算
- 某些策略（如动量策略）需要较长的历史数据来计算信号
- 凯利策略可能产生较为激进的配置，建议在实际应用中适当调整参数

## 贡献指南

欢迎提交 Pull Request 来改进代码或添加新功能。在提交之前，请确保：

1. 代码符合项目的编码规范
2. 添加了适当的单元测试
3. 更新了相关文档

## 许可证

MIT License 