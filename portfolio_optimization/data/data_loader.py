from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, Optional

class DataLoader(ABC):
    """数据加载基类"""
    
    @abstractmethod
    def load_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载数据
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            返回价格数据和收益率数据
        """
        pass
    
    def get_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        计算收益率
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
            
        Returns
        -------
        pd.DataFrame
            收益率数据
        """
        return prices.pct_change()
    
    def validate_data(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None) -> bool:
        """
        验证数据有效性
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据
            
        Returns
        -------
        bool
            数据是否有效
        """
        if prices.isnull().any().any():
            return False
        
        if returns is not None and returns.isnull().any().any():
            return False
            
        return True 