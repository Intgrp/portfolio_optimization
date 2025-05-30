import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy

class MeanVarianceStrategy(BaseStrategy):
    """均值方差策略"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252):
        """
        初始化均值方差策略
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        lookback_period : int, optional
            回溯期长度，默认为252个交易日
        """
        super().__init__(prices, returns, lookback_period)
        
    def generate_weights(self, date: str, target_return: Optional[float] = None,
                        risk_aversion: float = 1.0) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        target_return : float, optional
            目标收益率，如果为None则使用风险厌恶系数
        risk_aversion : float, optional
            风险厌恶系数，默认为1.0
            
        Returns
        -------
        pd.Series
            投资组合权重
        """
        historical_data = self.get_historical_data(date)
        mean_returns = historical_data.mean() * 252
        cov_matrix = historical_data.cov() * 252
        
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if target_return is not None:
                return portfolio_risk
            else:
                return -portfolio_return + risk_aversion * portfolio_risk
                
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # 权重和为1
        ]
        
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target_return}
            )
            
        bounds = [(0.0, 1.0) for _ in range(len(self.assets))]  # 权重在0和1之间
        
        # 初始权重为等权重
        initial_weights = np.array([1.0/len(self.assets)] * len(self.assets))
        
        # 优化求解
        result = minimize(objective, initial_weights, method='SLSQP',
                       constraints=constraints, bounds=bounds)
        
        if not result.success:
            return pd.Series(initial_weights, index=self.assets)
            
        return pd.Series(result.x, index=self.assets)
    
    def calculate_efficient_frontier(self, date: str, n_points: int = 100) -> pd.DataFrame:
        """
        计算有效前沿
        
        Parameters
        ----------
        date : str
            当前日期
        n_points : int, optional
            有效前沿上的点数，默认为100
            
        Returns
        -------
        pd.DataFrame
            有效前沿数据，包含收益率和风险
        """
        historical_data = self.get_historical_data(date)
        mean_returns = historical_data.mean() * 252
        cov_matrix = historical_data.cov() * 252
        
        # 计算最小和最大可能收益率
        min_return = min(mean_returns)
        max_return = max(mean_returns)
        
        # 生成目标收益率序列
        target_returns = np.linspace(min_return, max_return, n_points)
        efficient_frontier = []
        
        for target_return in target_returns:
            weights = self.generate_weights(date, target_return=target_return)
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
            efficient_frontier.append({
                '收益率': portfolio_return,
                '风险': portfolio_risk
            })
            
        return pd.DataFrame(efficient_frontier) 