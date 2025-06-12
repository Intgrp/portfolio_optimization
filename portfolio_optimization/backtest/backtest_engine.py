import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from portfolio_optimization.strategies.base_strategy import BaseStrategy

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None):
        """
        初始化回测引擎
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        """
        self.prices = prices
        self.returns = returns if returns is not None else prices.pct_change()
        
    def run_backtest(self, strategy: BaseStrategy, start_date: str, end_date: str,
                    rebalance_freq: str = 'M', **strategy_params) -> Tuple[pd.Series, pd.DataFrame]:
        """
        运行回测
        
        Parameters
        ----------
        strategy : BaseStrategy
            策略对象
        start_date : str
            开始日期
        end_date : str
            结束日期
        rebalance_freq : str, optional
            再平衡频率，默认为'M'（月度）
        **strategy_params : dict
            策略参数
            
        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            组合净值和权重历史
        """
        # 获取真实交易日
        trade_dates = self.prices.index[(self.prices.index >= start_date) & (self.prices.index <= end_date)]
        # 以周期分组，取每组最后一个交易日作为再平衡日
        trade_dates_df = pd.DataFrame(index=trade_dates)
        rebalance_dates = trade_dates_df.resample(rebalance_freq).last().index

        # 初始化权重历史
        weights_history = pd.DataFrame(index=self.prices.index, columns=self.prices.columns)
        current_weights = None

        for i, date in enumerate(trade_dates[:-1]):
            if date in rebalance_dates:
                current_weights = strategy.generate_weights(date)
            weights_history.loc[trade_dates[i+1]] = current_weights

        portfolio_returns = (self.returns * weights_history).sum(axis=1)
        portfolio_values = (1 + portfolio_returns).cumprod()

        return portfolio_values, weights_history
    
    def run_multiple_backtests(self, strategies: Dict[str, Tuple[BaseStrategy, Dict[str, Any]]],
                             start_date: str, end_date: str,
                             rebalance_freq: str = 'M') -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        运行多个策略的回测
        
        Parameters
        ----------
        strategies : Dict[str, Tuple[BaseStrategy, Dict[str, Any]]]
            策略字典，键为策略名称，值为(策略对象, 策略参数)元组
        start_date : str
            开始日期
        end_date : str
            结束日期
        rebalance_freq : str, optional
            再平衡频率，默认为'M'（月度）
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]
            所有策略的净值和权重历史
        """
        portfolio_values = pd.DataFrame()
        weights_history = {}
        
        for strategy_name, (strategy, params) in strategies.items():
            print(f"策略：{strategy} 执行回测")
            values, weights = self.run_backtest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                rebalance_freq=rebalance_freq,
                **params
            )
            portfolio_values[strategy_name] = values
            weights_history[strategy_name] = weights
            
        return portfolio_values, weights_history
    
    def calculate_performance_metrics(self, portfolio_values: pd.Series) -> Dict[str, float]:
        """
        计算策略表现指标
        
        Parameters
        ----------
        portfolio_values : pd.Series
            组合净值
            
        Returns
        -------
        Dict[str, float]
            策略表现指标
        """
        returns = portfolio_values.pct_change().dropna()
        
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        max_drawdown = (portfolio_values / portfolio_values.cummax() - 1).min()
        
        return {
            '年化收益率': annual_return,
            '年化波动率': annual_volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown
        }
    
    def compare_strategies(self, portfolio_values: pd.DataFrame) -> pd.DataFrame:
        """
        比较多个策略的表现
        
        Parameters
        ----------
        portfolio_values : pd.DataFrame
            多个策略的净值数据
            
        Returns
        -------
        pd.DataFrame
            策略表现对比
        """
        performance_metrics = {}
        
        for strategy in portfolio_values.columns:
            metrics = self.calculate_performance_metrics(portfolio_values[strategy])
            performance_metrics[strategy] = metrics
            
        return pd.DataFrame(performance_metrics).T 