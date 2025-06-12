import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy

class EqualWeightStrategy(BaseStrategy):
    """等权重策略：所有资产权重相等，不随收益率变化"""
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None, lookback_period: int = 252):
        super().__init__(prices, returns, lookback_period)

    def generate_weights(self, date: str, **kwargs) -> pd.Series:
        n_assets = len(self.assets)
        weights = pd.Series([1.0 / n_assets] * n_assets, index=self.assets)
        return weights 