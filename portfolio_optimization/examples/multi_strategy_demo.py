import sys
import os

import matplotlib.pyplot as plt
import pandas as pd

from portfolio_optimization.data.csv_loader import CsvDataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from portfolio_optimization.data.random_generator import RandomDataGenerator
from portfolio_optimization.strategies.mean_variance import MeanVarianceStrategy
from portfolio_optimization.strategies.risk_parity import RiskParityStrategy
from portfolio_optimization.strategies.conditional_risk_parity import ConditionalRiskParityStrategy
from portfolio_optimization.strategies.momentum_strategies import MaximumMomentumStrategy, ThresholdMomentumStrategy
from portfolio_optimization.strategies.hierarchical_strategies import HierarchicalRaffinotStrategy, HierarchicalMomentumStrategy
from portfolio_optimization.strategies.kelly_strategy import KellyStrategy
from portfolio_optimization.strategies.equal_weight_strategy import EqualWeightStrategy
from portfolio_optimization.backtest.backtest_engine import BacktestEngine
from portfolio_optimization.visualization.performance_plots import PerformancePlotter

def main():
    # 定义资产列表
    assets = ['AP', 'CF', 'FG', 'IC', 'IF', 'IH', 'MA', 'OI', 'PF', 'PK', 'RM',
              'SA', 'SF', 'SM', 'SR', 'T', 'TA', 'TF', 'TS', 'UR', 'a', 'ag',
              'al', 'au', 'b', 'bu', 'c', 'cs', 'cu', 'eb', 'eg',
              'fu', 'hc', 'i', 'j', 'jd', 'jm', 'l', 'lh', 'lu', 'm', 'ni',
              'p', 'pb', 'pg', 'pp', 'rb', 'ru', 'sc', 'sn', 'sp', 'ss', 'v',
              'y', 'zn']
    
    # 设置日期范围（确保使用工作日）
    backtest_start = pd.Timestamp('2017-01-03').to_pydatetime()
    backtest_end = pd.Timestamp('2024-12-31').to_pydatetime()
    # 生成模拟数据
    # data_loader = RandomDataGenerator(assets=assets, seed=42)
    data_loader = CsvDataLoader(data_path=r"D:\workspace\temp\section_ml\data", assets=assets)
    prices, returns = data_loader.load_data(
        start_date=backtest_start.strftime('%Y-%m-%d'),
        end_date=backtest_end.strftime('%Y-%m-%d')
    )
    returns = returns[assets]
    prices = prices[assets]
    # prices = prices.iloc[1:,:]
    
    # 初始化策略
    strategies = {
        '等权重策略': (
            EqualWeightStrategy(prices=prices, returns=returns),
            {}
        ),
        # '均值方差策略': (
        #     MeanVarianceStrategy(prices=prices, returns=returns),
        #     {'risk_aversion': 1.0}
        # ),
        '风险平价策略': (
            RiskParityStrategy(prices=prices, returns=returns),
            {}
        ),
        '条件风险平价策略': (
            ConditionalRiskParityStrategy(prices=prices, returns=returns),
            {}
        ),
        # '最大动量策略': (
        #     MaximumMomentumStrategy(prices=prices, returns=returns),
        #     {'top_n': 8}
        # ),
        '动量阈值策略': (
            ThresholdMomentumStrategy(prices=prices, returns=returns),
            {'threshold': 0.0001}
        ),
        '分层拉菲诺策略': (
            HierarchicalRaffinotStrategy(prices=prices, returns=returns),
            {}
        ),
        # '层级动量策略': (
        #     HierarchicalMomentumStrategy(prices=prices, returns=returns),
        #     {'top_n_per_cluster': 2}
        # ),
        '凯利策略': (
            KellyStrategy(prices=prices, returns=returns, lookback_period=252),
            {}
        ),
    }
    
    # 初始化回测引擎
    backtest_engine = BacktestEngine(prices=prices, returns=returns)
    
    # 运行回测
    portfolio_values, weights_history = backtest_engine.run_multiple_backtests(
        strategies=strategies,
        start_date=backtest_start.strftime('%Y-%m-%d'),
        end_date=backtest_end.strftime('%Y-%m-%d'),
        rebalance_freq='M'
    )
    
    # 计算策略表现
    performance_comparison = backtest_engine.compare_strategies(portfolio_values)
    print("\n=== 策略表现对比 ===")
    print(performance_comparison)
    
    # 初始化可视化工具
    plotter = PerformancePlotter()
    
    # 绘制累计收益对比图
    plotter.plot_cumulative_returns(portfolio_values)
    
    # 绘制回撤对比图
    plotter.plot_drawdown(portfolio_values)
    
    # 绘制滚动指标图
    plotter.plot_rolling_metrics(portfolio_values)
    
    # 绘制相关性热力图
    strategy_returns = portfolio_values.pct_change().dropna()
    plotter.plot_correlation_heatmap(strategy_returns)
    plt.show()
    
if __name__ == "__main__":
    main() 