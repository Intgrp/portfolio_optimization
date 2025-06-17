import pandas as pd
from typing import Optional
from .base_strategy import BaseStrategy

class EqualWeightStrategy(BaseStrategy):
    """等权重策略：所有资产权重相等，不随收益率变化"""
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None, lookback_period: int = 252):
        super().__init__(prices, returns, lookback_period)

    def generate_weights(self, date: str, **kwargs) -> pd.Series:
        # 取当前调仓日的价格
        if date not in self.prices.index:
            raise ValueError(f"日期 {date} 不在价格数据中")
        price_row = self.prices.loc[date]
        # 只对有价格（非NaN）的品种分配权重
        valid_assets = price_row.dropna().index.tolist()
        n_valid = len(valid_assets)
        weights = pd.Series(0.0, index=self.assets)
        if n_valid > 0:
            weights.loc[valid_assets] = 1.0 / n_valid
        return weights 