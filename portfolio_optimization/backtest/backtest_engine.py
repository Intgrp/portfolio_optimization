import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from portfolio_optimization.strategies.base_strategy import BaseStrategy
import os

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, returns: Optional[pd.DataFrame] = None, output_dir: str = None):
        """
        初始化回测引擎
        
        Parameters
        ----------
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        """
        self.returns = returns
        if output_dir is None:
            output_dir = os.getcwd()
        self.output_dir = output_dir

        
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
        # 初始化权重历史
        # 权重历史应包含所有可能出现的资产
        all_assets = self.returns.columns.tolist()
        weights_history = pd.DataFrame(index=self.returns.index, columns=all_assets)

        # 获取真实交易日
        trade_dates = self.returns.index[(self.returns.index >= start_date) & (self.returns.index <= end_date)]

        # 如果有交易日数据，为第一个交易日生成并设置初始权重
        if not trade_dates.empty:
            # 获取第一个交易日可用的资产：收益率非NaN且非0的品种
            initial_available_returns = self.returns.loc[:trade_dates[0]]
            initial_assets = initial_available_returns.columns[
                (initial_available_returns != 0).any() & (~initial_available_returns.isna()).any()
            ].tolist()
            
            # 如果没有可用的初始资产，则返回空权重系列
            if not initial_assets:
                return pd.Series(), pd.DataFrame(index=self.returns.index, columns=self.returns.columns)

            initial_weights = strategy.generate_weights(trade_dates[0], current_assets=initial_assets, **strategy_params)
            weights_history.loc[trade_dates[0], initial_weights.index] = initial_weights
            # Keep track of the assets for which weights were just generated
            last_generated_assets = initial_assets
        else:
            last_generated_assets = [] # No initial assets if no trade dates
            # 如果没有交易日，则返回空权重系列
            return pd.Series(), pd.DataFrame(index=self.returns.index, columns=self.returns.columns)
        
        # 以周期分组，取每组最后一个交易日作为再平衡日
        trade_dates_df = pd.DataFrame(index=trade_dates)
        rebalance_dates = trade_dates_df.resample(rebalance_freq).last().index.intersection(trade_dates)

        # 遍历交易日，在再平衡日或新增品种时生成并设置新权重
        for date in trade_dates:
            # 获取当前交易日可用的资产：收益率非NaN且非0的品种
            current_available_returns = self.returns.loc[:date]
            current_available_assets = current_available_returns.columns[
                (current_available_returns != 0).any() & (~current_available_returns.isna()).any()
            ].tolist()
            
            # Check if it's a rebalance date or if new assets have appeared since the last weight generation
            if date in rebalance_dates or set(current_available_assets) != set(last_generated_assets):
                current_weights = strategy.generate_weights(date, current_assets=current_available_assets, **strategy_params)
                weights_history.loc[date, current_weights.index] = current_weights
                last_generated_assets = current_available_assets

        # 前向填充权重，确保在再平衡日之间使用上一次的权重
        weights_history = weights_history.ffill()

        # 确保权重历史只包含回测期间的交易日
        weights_history = weights_history.loc[trade_dates]
        
        # 将NaN权重填充为0，表示该资产在该时期没有持仓
        weights_history = weights_history.fillna(0)

        portfolio_each_returns = (self.returns.loc[trade_dates] * weights_history)
        portfolio_returns = portfolio_each_returns.sum(axis=1)

        if self.output_dir:
            daily_return_file_path = os.path.join(self.output_dir, '加权后各品种每日收益率')
            os.makedirs(daily_return_file_path, exist_ok=True)
            daily_return_file = os.path.join(daily_return_file_path, f'{strategy.strategy_name}_returns.csv')
            portfolio_each_returns.to_csv(daily_return_file)
            print(f"已保存所有策略累计收益率表到 {daily_return_file}")

        portfolio_values = 1 + portfolio_returns.cumsum()

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
        
        if self.output_dir:
            weights_dir = os.path.join(self.output_dir, '权重')
            os.makedirs(weights_dir, exist_ok=True)
            result_dir = os.path.join(self.output_dir, '回测结果')
            os.makedirs(result_dir, exist_ok=True)

        for strategy_name, (strategy, params) in strategies.items():
            print(f"策略：{strategy_name} 开始回测")
            values, weights = self.run_backtest(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                rebalance_freq=rebalance_freq,
                **params
            )
            portfolio_values[strategy_name] = values
            weights_history[strategy_name] = weights
            
            if self.output_dir:
                weights_file_path = os.path.join(self.output_dir, '权重', f'{strategy_name}_weights.csv')
                weights.to_csv(weights_file_path)
                print(f"已保存 {strategy_name} 权重表到 {weights_file_path}")
        # 合并输出所有策略的累计收益率表
        if self.output_dir:
            returns_file_path = os.path.join(self.output_dir, '回测结果', 'all_strategies_cumulative_returns.csv')
            portfolio_values.to_csv(returns_file_path)
            print(f"已保存所有策略累计收益率表到 {returns_file_path}")

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

    def save_performance_report(self, performance_df: pd.DataFrame, output_dir: str, filename: str = "performance_report.csv"):
        """
        保存策略表现报告
        
        Parameters
        ----------
        performance_df : pd.DataFrame
            策略表现数据框
        output_dir : str
            输出文件夹路径
        filename : str, optional
            文件名，默认为"performance_report.csv"
        """
        if output_dir:
            result_dir = os.path.join(output_dir, '回测结果')
            os.makedirs(result_dir, exist_ok=True)
            file_path = os.path.join(result_dir, filename)
            performance_df.to_csv(file_path)
            print(f"已保存策略表现报告到 {file_path}") 