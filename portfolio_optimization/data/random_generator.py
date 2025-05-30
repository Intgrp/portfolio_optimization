import numpy as np
import pandas as pd
from typing import List, Tuple
from .data_loader import DataLoader

class RandomDataGenerator(DataLoader):
    """随机数据生成器"""
    
    def __init__(self, assets: List[str], seed: int = 42):
        """
        初始化随机数据生成器
        
        Parameters
        ----------
        assets : List[str]
            资产列表
        seed : int, optional
            随机种子，默认为42
        """
        self.assets = assets
        self.seed = seed
        np.random.seed(seed)
        
    def load_data(self, start_date: str, end_date: str, 
                 mu: float = 0.0002, sigma: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成随机数据
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        mu : float, optional
            收益率均值，默认为0.0002
        sigma : float, optional
            收益率标准差，默认为0.02
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            返回价格数据和收益率数据
        """
        # 生成日期序列
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_assets = len(self.assets)
        
        # 生成价格数据
        initial_prices = np.random.uniform(50, 200, n_assets)
        prices = pd.DataFrame(index=dates, columns=self.assets)
        prices.iloc[0] = initial_prices
        
        # 生成收益率数据
        returns_data = np.random.normal(mu, sigma, (len(dates)-1, n_assets))
        returns = pd.DataFrame(returns_data, index=dates[1:], columns=self.assets)
        
        # 根据收益率计算价格
        for i in range(1, len(dates)):
            prices.iloc[i] = prices.iloc[i-1] * (1 + returns.iloc[i-1])
            
        return prices, returns
        
    def generate_correlated_returns(self, start_date: str, end_date: str, 
                                  correlation_matrix: np.ndarray,
                                  mu: float = 0.0002, sigma: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        生成具有相关性的随机数据
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        correlation_matrix : np.ndarray
            相关系数矩阵
        mu : float, optional
            收益率均值，默认为0.0002
        sigma : float, optional
            收益率标准差，默认为0.02
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            返回价格数据和收益率数据
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n_assets = len(self.assets)
        
        # 生成相关的随机数
        L = np.linalg.cholesky(correlation_matrix)
        uncorrelated_returns = np.random.normal(0, 1, (len(dates)-1, n_assets))
        correlated_returns = mu + sigma * (uncorrelated_returns @ L.T)
        
        # 转换为DataFrame
        returns = pd.DataFrame(correlated_returns, index=dates[1:], columns=self.assets)
        
        # 生成价格数据
        initial_prices = np.random.uniform(50, 200, n_assets)
        prices = pd.DataFrame(index=dates, columns=self.assets)
        prices.iloc[0] = initial_prices
        
        # 根据收益率计算价格
        for i in range(1, len(dates)):
            prices.iloc[i] = prices.iloc[i-1] * (1 + returns.iloc[i-1])
            
        return prices, returns 