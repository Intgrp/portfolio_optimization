import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from .base_strategy import BaseStrategy

class KellyStrategy(BaseStrategy):
    """基于凯利公式的投资策略"""
    
    def __init__(self, prices: pd.DataFrame, returns: pd.DataFrame, 
                 lookback_period: int = 252):
        """
        初始化凯利策略
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame
            收益率数据
        lookback_period : int, optional
            回看期长度，默认为252个交易日（一年）
        """
        super().__init__(prices, returns)
        self.lookback_period = lookback_period
        self.strategy_name = "凯利策略"
        
    def calculate_kelly_fraction(self, historical_returns: pd.Series) -> float:
        """
        计算单个资产的凯利比例
        
        Parameters
        ----------
        historical_returns : pd.Series
            历史收益率序列
            
        Returns
        -------
        float
            凯利比例
        """
        mean_return = historical_returns.mean()
        var_return = historical_returns.var()
        
        if var_return == 0:
            return 0
            
        # 使用完整凯利公式：f* = μ/σ²
        kelly_fraction = mean_return / var_return
        
        # 对凯利比例进行限制，避免过度杠杆
        kelly_fraction = np.clip(kelly_fraction, -1, 1)
        
        return kelly_fraction
        
    def generate_weights(self, date: str, current_assets: Optional[List[str]] = None, **kwargs) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        current_assets : Optional[List[str]], optional
            当前可用的资产列表，如果为None则使用策略初始化时的所有资产
        
        Returns
        -------
        pd.Series
            资产权重序列
        """
        historical_data = self.get_historical_data(date, current_assets=current_assets)
        
        # If no current assets are provided, default to all assets known to the strategy
        if current_assets is None:
            current_assets = self.assets
            
        # Filter historical data to only include current_assets that have data up to the current date
        historical_data = historical_data[historical_data.columns.intersection(current_assets)]
        
        min_valid_days = int(self.lookback_period * 0.8)
        valid_assets = historical_data.columns[historical_data.notna().sum() > min_valid_days]
        filtered_data = historical_data[valid_assets]
        
        if len(valid_assets) == 0:
            return pd.Series(0, index=current_assets)
            
        mean_returns = filtered_data.mean()
        cov_matrix = filtered_data.cov()
        
        try:
            inv_cov = np.linalg.inv(cov_matrix.values)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_matrix.values)
            
        kelly_weights = inv_cov @ mean_returns.values
        kelly_weights = np.maximum(kelly_weights, 0)
        
        if kelly_weights.sum() > 0:
            kelly_weights = kelly_weights / kelly_weights.sum()
            
        # Create a Series with all current_assets and fill with zeros, then assign kelly_weights
        weights = pd.Series(0, index=current_assets)
        weights[valid_assets] = kelly_weights
        
        return weights 