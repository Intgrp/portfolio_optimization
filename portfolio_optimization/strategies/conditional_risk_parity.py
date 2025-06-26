import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional, Tuple, List
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
        self.strategy_name = "条件风险平价策略"
        
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
            return pd.Series(0, index=current_assets)
            
        cond_cov_matrix, _ = self.calculate_conditional_covariance(
            filtered_data, self.confidence_level
        )
        n_valid = len(valid_assets)
        initial_weights = np.array([1.0/n_valid] * n_valid)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(n_valid)]
        
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
            valid_weights = initial_weights
        else:
            valid_weights = result.x
        # Create a Series with all current_assets and fill with zeros, then assign valid_weights
        weights = pd.Series(0, index=current_assets)
        weights[valid_assets] = valid_weights
        
        return weights
    
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
        min_valid_days = int(self.lookback_period * 0.8)
        valid_assets = historical_data.columns[historical_data.notna().sum() > min_valid_days]
        filtered_data = historical_data[valid_assets]
        if len(valid_assets) == 0:
            return pd.Series(0, index=self.assets)
        cond_cov_matrix, _ = self.calculate_conditional_covariance(
            filtered_data, self.confidence_level
        )
        risk_contrib = self.calculate_conditional_risk_contribution(
            weights[valid_assets].values, cond_cov_matrix
        )
        result = pd.Series(0, index=self.assets)
        result[valid_assets] = risk_contrib
        return result 