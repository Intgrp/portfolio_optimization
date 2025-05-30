import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
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
        target_risk = np.sqrt(weights.T @ cov_matrix @ weights) / len(weights)
        return np.sum((risk_contrib - target_risk)**2)
        
    def generate_weights(self, date: str, **kwargs) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
            
        Returns
        -------
        pd.Series
            投资组合权重
        """
        historical_data = self.get_historical_data(date)
        cov_matrix = historical_data.cov() * 252
        
        n_assets = len(self.assets)
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # 权重和为1
        ]
        
        bounds = [(0.0, 1.0) for _ in range(n_assets)]  # 权重在0和1之间
        
        # 优化求解
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
            return pd.Series(initial_weights, index=self.assets)
            
        return pd.Series(result.x, index=self.assets)
        
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
        cov_matrix = historical_data.cov() * 252
        
        risk_contrib = self.calculate_risk_contribution(
            weights.values, cov_matrix
        )
        
        return pd.Series(risk_contrib, index=self.assets) 