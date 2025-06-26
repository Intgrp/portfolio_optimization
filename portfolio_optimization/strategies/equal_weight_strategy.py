import pandas as pd
from typing import Optional, List
from .base_strategy import BaseStrategy

class EqualWeightStrategy(BaseStrategy):
    """等权重策略：所有资产权重相等，不随收益率变化"""
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None, lookback_period: int = 252):
        super().__init__(prices, returns, lookback_period)
        self.strategy_name = "等权重策略"

    def generate_weights(self, date: str, current_assets: Optional[List[str]] = None, **kwargs) -> pd.Series:
        # 确定要分配权重的资产列表
        assets_to_allocate = current_assets if current_assets else self.assets

        if not assets_to_allocate:
            return pd.Series(dtype=float) # 如果没有资产，返回空Series

        n_assets = len(assets_to_allocate)
        # 为所有当前可用的资产分配等权重
        weights = pd.Series(1.0 / n_assets, index=assets_to_allocate)

        return weights 