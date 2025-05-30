import numpy as np
import pandas as pd
from typing import Optional, List
from .base_strategy import BaseStrategy

class MomentumStrategyBase(BaseStrategy):
    """动量策略基类"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252, momentum_periods: List[int] = None,
                 momentum_weights: List[float] = None):
        """
        初始化动量策略基类
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        lookback_period : int, optional
            回溯期长度，默认为252个交易日
        momentum_periods : List[int], optional
            动量计算期长度列表，默认为[21, 63, 252]（1个月、3个月、1年）
        momentum_weights : List[float], optional
            动量权重列表，默认为[0.5, 0.3, 0.2]
        """
        super().__init__(prices, returns, lookback_period)
        self.momentum_periods = momentum_periods or [21, 63, 252]
        self.momentum_weights = momentum_weights or [0.5, 0.3, 0.2]
        
    def calculate_momentum_score(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        计算动量得分
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            历史数据
            
        Returns
        -------
        pd.Series
            动量得分
        """
        momentum_components = []
        
        for period in self.momentum_periods:
            if len(historical_data) < period:
                period = len(historical_data)
            momentum_components.append(historical_data.iloc[-period:].mean())
            
        momentum_score = sum(w * m for w, m in zip(self.momentum_weights, momentum_components))
        return momentum_score

class MaximumMomentumStrategy(MomentumStrategyBase):
    """最大动量策略"""
    
    def generate_weights(self, date: str, top_n: int = 8, **kwargs) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        top_n : int, optional
            选择动量最大的前n个资产，默认为8
            
        Returns
        -------
        pd.Series
            投资组合权重
        """
        historical_data = self.get_historical_data(date)
        momentum_scores = self.calculate_momentum_score(historical_data)
        
        # 选择动量最大的top_n个资产
        top_assets = momentum_scores.nlargest(top_n).index
        
        # 生成权重
        weights = pd.Series(0.0, index=self.assets)
        weights[top_assets] = 1.0 / top_n
        
        return weights

class ThresholdMomentumStrategy(MomentumStrategyBase):
    """动量阈值策略"""
    
    def generate_weights(self, date: str, threshold: float = 0.0, **kwargs) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        threshold : float, optional
            动量阈值，默认为0.0
            
        Returns
        -------
        pd.Series
            投资组合权重
        """
        historical_data = self.get_historical_data(date)
        momentum_scores = self.calculate_momentum_score(historical_data)
        
        # 选择动量大于阈值的资产
        selected_assets = momentum_scores[momentum_scores > threshold].index
        
        # 如果没有资产超过阈值，则使用等权重
        if len(selected_assets) == 0:
            return pd.Series(1.0/len(self.assets), index=self.assets)
            
        # 生成权重
        weights = pd.Series(0.0, index=self.assets)
        weights[selected_assets] = 1.0 / len(selected_assets)
        
        return weights 