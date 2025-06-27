import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from typing import Optional, Dict, List, Tuple
from .base_strategy import BaseStrategy
from .momentum_strategies import MomentumStrategyBase

class HierarchicalStrategyBase(BaseStrategy):
    """层级策略基类"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252, n_clusters: int = 8):
        """
        初始化层级策略基类
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        lookback_period : int, optional
            回溯期长度，默认为252个交易日
        n_clusters : int, optional
            聚类数量，默认为8
        """
        super().__init__(prices, returns, lookback_period)
        self.n_clusters = n_clusters
        self.strategy_name = "层级策略基类"
        
    def calculate_distance_matrix(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算距离矩阵
        
        Parameters
        ----------
        historical_data : pd.DataFrame
            历史数据
            
        Returns
        -------
        pd.DataFrame
            距离矩阵
        """
        corr_matrix = historical_data.corr()
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        return pd.DataFrame(distance_matrix, index=historical_data.columns, columns=historical_data.columns)
        
    def perform_clustering(self, distance_matrix: pd.DataFrame) -> np.ndarray:
        """
        执行层级聚类
        
        Parameters
        ----------
        distance_matrix : pd.DataFrame
            距离矩阵
            
        Returns
        -------
        np.ndarray
            聚类结果
        """
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')
        clusters = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust')
        return clusters
        
    def get_cluster_assets(self, clusters: np.ndarray, valid_assets: list) -> Dict[int, List[str]]:
        """
        获取每个簇的资产列表
        
        Parameters
        ----------
        clusters : np.ndarray
            聚类结果
        valid_assets : list
            有效资产列表
            
        Returns
        -------
        Dict[int, List[str]]
            每个簇的资产列表
        """
        cluster_assets = {}
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_assets[cluster_id] = [asset for i, asset in enumerate(valid_assets)
                                        if clusters[i] == cluster_id]
        return cluster_assets

class HierarchicalRaffinotStrategy(HierarchicalStrategyBase):
    """分层拉菲诺策略"""
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None, lookback_period: int = 252):
        super().__init__(prices, returns, lookback_period)
        self.strategy_name = "分层拉菲诺策略"
    
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
        valid_assets = historical_data.columns[historical_data.notna().sum() > min_valid_days].tolist()
        filtered_data = historical_data[valid_assets]
        
        if len(valid_assets) == 0:
            return pd.Series(0, index=current_assets)
            
        distance_matrix = self.calculate_distance_matrix(filtered_data)
        clusters = self.perform_clustering(distance_matrix)
        cluster_assets = self.get_cluster_assets(clusters, valid_assets)
        cov_matrix = filtered_data.cov() * 252
        
        # Initialize weights with all current_assets, filled with zeros
        weights = pd.Series(0.0, index=current_assets)
        
        for cluster_id, assets in cluster_assets.items():
            if not assets:
                continue
            cluster_cov = cov_matrix.loc[assets, assets]
            n_assets = len(assets)
            initial_weights = np.array([1.0/n_assets] * n_assets)
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
            ]
            bounds = [(0.0, 1.0) for _ in range(n_assets)]
            result = minimize(
                fun=self.risk_parity_objective,
                x0=initial_weights,
                args=(cluster_cov,),
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'ftol': 1e-12}
            )
            if result.success:
                cluster_weights = pd.Series(result.x, index=assets)
            else:
                cluster_weights = pd.Series(initial_weights, index=assets)
            weights[assets] = cluster_weights * (1.0 / self.n_clusters)
            
        return weights

class HierarchicalMomentumStrategy(HierarchicalStrategyBase, MomentumStrategyBase):
    """层级动量策略"""
    
    def __init__(self, prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None,
                 lookback_period: int = 252, n_clusters: int = 8,
                 momentum_periods: List[int] = None, momentum_weights: List[float] = None):
        """
        初始化层级动量策略
        
        Parameters
        ----------
        prices : pd.DataFrame
            价格数据
        returns : pd.DataFrame, optional
            收益率数据，如果为None则根据价格数据计算
        lookback_period : int, optional
            回溯期长度，默认为252个交易日
        n_clusters : int, optional
            聚类数量，默认为8
        momentum_periods : List[int], optional
            动量计算期长度列表
        momentum_weights : List[float], optional
            动量权重列表
        """
        HierarchicalStrategyBase.__init__(self, prices, returns, lookback_period, n_clusters)
        MomentumStrategyBase.__init__(self, prices, returns, lookback_period,
                                    momentum_periods, momentum_weights)
        self.strategy_name = "层级动量策略"
        
    def generate_weights(self, date: str, current_assets: Optional[List[str]] = None, top_n_per_cluster: int = 2, **kwargs) -> pd.Series:
        """
        生成投资组合权重
        
        Parameters
        ----------
        date : str
            当前日期
        current_assets : Optional[List[str]], optional
            当前可用的资产列表，如果为None则使用策略初始化时的所有资产
        top_n_per_cluster : int, optional
            每个簇选择的资产数量，默认为2
            
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
        valid_assets = historical_data.columns[historical_data.notna().sum() > min_valid_days].tolist()
        filtered_data = historical_data[valid_assets]
        
        if len(valid_assets) == 0:
            return pd.Series(0, index=current_assets)

        distance_matrix = self.calculate_distance_matrix(filtered_data)
        clusters = self.perform_clustering(distance_matrix)
        cluster_assets = self.get_cluster_assets(clusters, valid_assets)
        
        # 计算动量得分
        momentum_scores = self.calculate_momentum_score(filtered_data)
        
        # 初始化权重，确保索引包含所有current_assets
        weights = pd.Series(0.0, index=current_assets)
        
        # 对每个簇选择动量最大的资产
        print(f"层级聚类下，划分的簇个数：{len(cluster_assets)}")
        for cluster_id, assets in cluster_assets.items():
            if not assets:
                continue
                
            # 选择簇内动量最大的top_n个资产
            cluster_scores = momentum_scores[assets]
            top_assets = cluster_scores.nlargest(
                min(top_n_per_cluster, len(assets))
            ).index
            
            # 对选中的资产等权重配置
            if len(top_assets) > 0:
                weights[top_assets] = 1.0 / (self.n_clusters * len(top_assets))
            
        return weights 