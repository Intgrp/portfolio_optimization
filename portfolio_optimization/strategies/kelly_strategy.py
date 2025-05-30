import numpy as np
import pandas as pd
from typing import Dict, Optional
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
        
    def generate_weights(self, date: str, **kwargs) -> Dict[str, float]:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        
        Returns
        -------
        Dict[str, float]
            资产权重字典
        """
        # 获取历史数据
        historical_data = self.get_historical_data(date)
        
        if len(historical_data) < self.lookback_period:
            return {asset: 1.0/len(self.assets) for asset in self.assets}
            
        # 使用最近的lookback_period数据
        recent_data = historical_data.iloc[-self.lookback_period:]
        
        # 计算每个资产的凯利比例
        kelly_fractions = {}
        for asset in self.assets:
            kelly_fractions[asset] = self.calculate_kelly_fraction(recent_data[asset])
            
        # 将负的凯利比例设为0（不做空）
        kelly_fractions = {k: max(0, v) for k, v in kelly_fractions.items()}
        
        # 如果所有资产的凯利比例都为0，则平均分配
        if sum(kelly_fractions.values()) == 0:
            return {asset: 1.0/len(self.assets) for asset in self.assets}
            
        # 归一化权重
        total_fraction = sum(kelly_fractions.values())
        weights = {asset: fraction/total_fraction for asset, fraction in kelly_fractions.items()}
        
        return weights 