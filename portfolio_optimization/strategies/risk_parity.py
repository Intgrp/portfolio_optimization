import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, List
from .base_strategy import BaseStrategy

class RiskParityStrategy(BaseStrategy):
    """风险平价策略"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252):
        """
        初始化风险平价策略
        
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
        self.strategy_name = "风险平价策略"
        
    def calculate_risk_contribution(self, weights: np.ndarray,
                                 cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        计算风险贡献
        
        Parameters
        ----------
        weights : np.ndarray
            投资组合权重
        cov_matrix : pd.DataFrame
            协方差矩阵
            
        Returns
        -------
        np.ndarray
            风险贡献
        """
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_risk = cov_matrix @ weights
        risk_contribution = weights * marginal_risk / portfolio_vol
        return risk_contribution
        
    def risk_parity_objective(self, weights: np.ndarray,
                            cov_matrix: pd.DataFrame) -> float:
        """
        风险平价目标函数
        
        Parameters
        ----------
        weights : np.ndarray
            投资组合权重
        cov_matrix : pd.DataFrame
            协方差矩阵
            
        Returns
        -------
        float
            目标函数值
        """
        risk_contrib = self.calculate_risk_contribution(weights, cov_matrix)
        mean_rc = np.mean(risk_contrib)
        return np.sum((risk_contrib - mean_rc)**2)
        
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
            投资组合权重
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
            # If no valid assets, return a series of zeros for current_assets
            return pd.Series(0, index=current_assets)
            
        cov_matrix = filtered_data.cov() * 252
        n_valid = len(valid_assets)
        initial_weights = np.array([1.0/n_valid] * n_valid)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(n_valid)]
        
        result = minimize(
            fun=self.risk_parity_objective,
            x0=initial_weights,
            args=(cov_matrix,),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-12}
        )
        
        if not result.success:
            valid_weights = initial_weights
        else:
            valid_weights = result.x
            
        # Create a Series with all current_assets and fill with zeros, then assign valid_weights
        weights = pd.Series(0, index=current_assets)
        weights[valid_assets] = valid_weights
        
        return weights
        
    def get_risk_contributions(self, weights: pd.Series, date: str) -> pd.Series:
        """
        获取风险贡献
        
        Parameters
        ----------
        weights : pd.Series
            投资组合权重
        date : str
            当前日期
            
        Returns
        -------
        pd.Series
            各资产的风险贡献
        """
        historical_data = self.get_historical_data(date)
        min_valid_days = int(self.lookback_period * 0.8)
        valid_assets = historical_data.columns[historical_data.notna().sum() > min_valid_days]
        filtered_data = historical_data[valid_assets]
        if len(valid_assets) == 0:
            return pd.Series(0, index=self.assets)
        cov_matrix = filtered_data.cov() * 252
        risk_contrib = self.calculate_risk_contribution(
            weights[valid_assets].values, cov_matrix
        )
        result = pd.Series(0, index=self.assets)
        result[valid_assets] = risk_contrib
        return result 