import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple
from .base_strategy import BaseStrategy

class ConditionalRiskParityStrategy(BaseStrategy):
    """条件风险平价策略"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252, confidence_level: float = 0.95):
        """
        初始化条件风险平价策略
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        lookback_period : int, optional
            回溯期长度，默认为252个交易日
        confidence_level : float, optional
            置信水平，默认为0.95
        """
        super().__init__(prices, returns, lookback_period)
        self.confidence_level = confidence_level
        
    def calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """
        计算条件风险价值（CVaR）
        
        Parameters
        ----------
        returns : pd.Series
            收益率序列
        confidence_level : float
            置信水平
            
        Returns
        -------
        float
            CVaR值
        """
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var
    
    def calculate_conditional_covariance(self, returns: pd.DataFrame, 
                                      confidence_level: float) -> Tuple[pd.DataFrame, pd.Series]:
        """
        计算条件协方差矩阵
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        confidence_level : float
            置信水平
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            条件协方差矩阵和条件均值
        """
        # 计算每个资产的VaR
        vars = {col: np.percentile(returns[col], (1 - confidence_level) * 100)
               for col in returns.columns}
        
        # 选择尾部事件
        tail_events = returns.apply(lambda x: x <= vars[x.name])
        tail_returns = returns[tail_events.any(axis=1)]
        
        if len(tail_returns) == 0:
            return returns.cov(), returns.mean()
            
        return tail_returns.cov(), tail_returns.mean()
    
    def calculate_conditional_risk_contribution(self, weights: np.ndarray,
                                             cond_cov_matrix: pd.DataFrame) -> np.ndarray:
        """
        计算条件风险贡献
        
        Parameters
        ----------
        weights : np.ndarray
            投资组合权重
        cond_cov_matrix : pd.DataFrame
            条件协方差矩阵
            
        Returns
        -------
        np.ndarray
            条件风险贡献
        """
        portfolio_vol = np.sqrt(weights.T @ cond_cov_matrix @ weights)
        marginal_risk = cond_cov_matrix @ weights
        risk_contribution = weights * marginal_risk / portfolio_vol
        return risk_contribution
    
    def conditional_risk_parity_objective(self, weights: np.ndarray,
                                        cond_cov_matrix: pd.DataFrame) -> float:
        """
        条件风险平价目标函数
        
        Parameters
        ----------
        weights : np.ndarray
            投资组合权重
        cond_cov_matrix : pd.DataFrame
            条件协方差矩阵
            
        Returns
        -------
        float
            目标函数值
        """
        risk_contrib = self.calculate_conditional_risk_contribution(weights, cond_cov_matrix)
        target_risk = np.sqrt(weights.T @ cond_cov_matrix @ weights) / len(weights)
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
        cond_cov_matrix, _ = self.calculate_conditional_covariance(
            historical_data, self.confidence_level
        )
        
        n_assets = len(self.assets)
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # 权重和为1
        ]
        
        bounds = [(0.0, 1.0) for _ in range(n_assets)]  # 权重在0和1之间
        
        # 优化求解
        result = minimize(
            fun=self.conditional_risk_parity_objective,
            x0=initial_weights,
            args=(cond_cov_matrix,),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-12}
        )
        
        if not result.success:
            return pd.Series(initial_weights, index=self.assets)
            
        return pd.Series(result.x, index=self.assets)
    
    def get_conditional_risk_contributions(self, weights: pd.Series, 
                                        date: str) -> pd.Series:
        """
        获取条件风险贡献
        
        Parameters
        ----------
        weights : pd.Series
            投资组合权重
        date : str
            当前日期
            
        Returns
        -------
        pd.Series
            各资产的条件风险贡献
        """
        historical_data = self.get_historical_data(date)
        cond_cov_matrix, _ = self.calculate_conditional_covariance(
            historical_data, self.confidence_level
        )
        
        risk_contrib = self.calculate_conditional_risk_contribution(
            weights.values, cond_cov_matrix
        )
        
        return pd.Series(risk_contrib, index=self.assets) 