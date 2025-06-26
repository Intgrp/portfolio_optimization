from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252):
        """
        初始化策略
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        lookback_period : int, optional
            回溯期长度，默认为252个交易日
        """
        self.strategy_name = "基类策略"
        self.prices = prices
        self.returns = returns if returns is not None else prices.pct_change()
        self.lookback_period = lookback_period
        self.assets = prices.columns.tolist()
        
    @abstractmethod
    def generate_weights(self, date: str, current_assets: Optional[List[str]] = None, **kwargs) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        current_assets : Optional[List[str]], optional
            当前可用的资产列表，如果为None则使用策略初始化时的所有资产
        **kwargs : dict
            其他参数
            
        Returns
        -------
        pd.Series
            投资组合权重
        """
        pass
    
    def get_historical_data(self, date: str, current_assets: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取历史数据
        
        Parameters
        ----------
        date : str
            当前日期
        current_assets : Optional[List[str]], optional
            当前可用的资产列表，如果提供，则只返回这些资产的历史数据
            
        Returns
        -------
        pd.DataFrame
            历史数据
        """
        date_loc = self.returns.index.get_loc(date)
        start_loc = max(0, date_loc - self.lookback_period + 1)
        
        historical_returns = self.returns.iloc[start_loc:date_loc + 1]
        
        if current_assets:
            # Ensure only available assets are considered, and handle cases where an asset might not have historical data yet
            historical_returns = historical_returns[historical_returns.columns.intersection(current_assets)]
            
        return historical_returns
    
    def calculate_portfolio_metrics(self, weights: pd.Series, 
                                  historical_data: pd.DataFrame) -> Dict[str, float]:
        """
        计算投资组合指标
        
        Parameters
        ----------
        weights : pd.Series
            投资组合权重
        historical_data : pd.DataFrame
            历史数据
            
        Returns
        -------
        Dict[str, float]
            投资组合指标
        """
        portfolio_return = (historical_data * weights).sum(axis=1)
        portfolio_std = portfolio_return.std() * np.sqrt(252)
        portfolio_mean = portfolio_return.mean() * 252
        sharpe_ratio = portfolio_mean / portfolio_std if portfolio_std != 0 else 0
        
        return {
            'return': portfolio_mean,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def validate_weights(self, weights: pd.Series) -> bool:
        """
        验证权重有效性
        
        Parameters
        ----------
        weights : pd.Series
            投资组合权重
            
        Returns
        -------
        bool
            权重是否有效
        """
        if not np.isclose(weights.sum(), 1.0, rtol=1e-5):
            return False
            
        if (weights < 0).any():
            return False
            
        return True 