import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Optional
import os

class PerformancePlotter:
    """绩效可视化类"""
    
    def __init__(self, output_dir: str=None):
        """初始化绩效可视化类"""
        plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.output_dir = output_dir

        
    def plot_cumulative_returns(self, portfolio_values: pd.DataFrame,
                              title: str = '策略累计收益对比',
                              figsize: tuple = (15, 8),
                              filename: str = 'cumulative_returns.png') -> None:
        """
        绘制累计收益对比图
        
        Parameters
        ----------
        portfolio_values : pd.DataFrame
            策略净值数据
        title : str, optional
            图表标题，默认为'策略累计收益对比'
        figsize : tuple, optional
            图表大小，默认为(15, 8)
        filename : str, optional
            文件名，默认为'cumulative_returns.png'
        """
        plt.figure(figsize=figsize)
        for strategy in portfolio_values.columns:
            plt.plot(portfolio_values.index, portfolio_values[strategy], 
                    label=strategy, linewidth=2)
            
        plt.title(title, fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累计收益', fontsize=12)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if self.output_dir:
            img_dir = os.path.join(self.output_dir, '图片')
            os.makedirs(img_dir, exist_ok=True)
            file_path = os.path.join(img_dir, filename)
            plt.savefig(file_path)
            print(f"已保存累计收益图到 {file_path}")
        plt.close()
        
    def plot_drawdown(self, portfolio_values: pd.DataFrame,
                     title: str = '策略回撤对比',
                     figsize: tuple = (15, 8),
                     filename: str = 'drawdown.png') -> None:
        """
        绘制回撤对比图
        
        Parameters
        ----------
        portfolio_values : pd.DataFrame
            策略净值数据
        title : str, optional
            图表标题，默认为'策略回撤对比'
        figsize : tuple, optional
            图表大小，默认为(15, 8)
        filename : str, optional
            文件名，默认为'drawdown.png'
        """
        plt.figure(figsize=figsize)
        for strategy in portfolio_values.columns:
            drawdown = (portfolio_values[strategy] / portfolio_values[strategy].cummax() - 1)
            plt.plot(portfolio_values.index, drawdown, 
                    label=strategy, linewidth=2)
            
        plt.title(title, fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('回撤', fontsize=12)
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if self.output_dir:
            img_dir = os.path.join(self.output_dir, '图片')
            os.makedirs(img_dir, exist_ok=True)
            file_path = os.path.join(img_dir, filename)
            plt.savefig(file_path)
            print(f"已保存回撤图到 {file_path}")
        plt.close()
        
    def plot_rolling_metrics(self, portfolio_values: pd.DataFrame,
                           window: int = 252,
                           metrics: Optional[list] = None,
                           figsize: tuple = (15, 15),
                           filename: str = 'rolling_metrics.png') -> None:
        """
        绘制滚动指标图
        
        Parameters
        ----------
        portfolio_values : pd.DataFrame
            策略净值数据
        window : int, optional
            滚动窗口长度，默认为252
        metrics : list, optional
            需要计算的指标列表，默认为['收益率', '波动率', '夏普比率']
        figsize : tuple, optional
            图表大小，默认为(15, 15)
        filename : str, optional
            文件名，默认为'rolling_metrics.png'
        """
        if metrics is None:
            metrics = ['收益率', '波动率', '夏普比率']
            
        returns = portfolio_values.pct_change()
        n_metrics = len(metrics)
        
        plt.figure(figsize=figsize)
        for i, metric in enumerate(metrics, 1):
            plt.subplot(n_metrics, 1, i)
            
            if metric == '收益率':
                for strategy in returns.columns:
                    rolling_return = returns[strategy].rolling(window).mean() * 252
                    plt.plot(returns.index, rolling_return, 
                            label=strategy, linewidth=2)
                plt.title(f'滚动年化收益率 (窗口={window}天)', fontsize=12)
                    
            elif metric == '波动率':
                for strategy in returns.columns:
                    rolling_vol = returns[strategy].rolling(window).std() * np.sqrt(252)
                    plt.plot(returns.index, rolling_vol, 
                            label=strategy, linewidth=2)
                plt.title(f'滚动年化波动率 (窗口={window}天)', fontsize=12)
                    
            elif metric == '夏普比率':
                for strategy in returns.columns:
                    rolling_return = returns[strategy].rolling(window).mean() * 252
                    rolling_vol = returns[strategy].rolling(window).std() * np.sqrt(252)
                    rolling_sharpe = rolling_return / rolling_vol
                    plt.plot(returns.index, rolling_sharpe, 
                            label=strategy, linewidth=2)
                plt.title(f'滚动夏普比率 (窗口={window}天)', fontsize=12)
                
            plt.xlabel('日期', fontsize=10)
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
        plt.tight_layout()
        
        if self.output_dir:
            img_dir = os.path.join(self.output_dir, '图片')
            os.makedirs(img_dir, exist_ok=True)
            file_path = os.path.join(img_dir, filename)
            plt.savefig(file_path)
            print(f"已保存滚动指标图到 {file_path}")
        plt.close()
        
    def plot_correlation_heatmap(self, returns: pd.DataFrame,
                               title: str = '策略相关性热力图',
                               figsize: tuple = (10, 8),
                               filename: str = 'correlation_heatmap.png') -> None:
        """
        绘制相关性热力图
        
        Parameters
        ----------
        returns : pd.DataFrame
            收益率数据
        title : str, optional
            图表标题，默认为'策略相关性热力图'
        figsize : tuple, optional
            图表大小，默认为(10, 8)
        filename : str, optional
            文件名，默认为'correlation_heatmap.png'
        """
        plt.figure(figsize=figsize)
        correlation = returns.corr()
        
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True)
        plt.title(title, fontsize=14)
        plt.tight_layout()
        
        if self.output_dir:
            img_dir = os.path.join(self.output_dir, '图片')
            os.makedirs(img_dir, exist_ok=True)
            file_path = os.path.join(img_dir, filename)
            plt.savefig(file_path)
            print(f"已保存相关性热力图到 {file_path}")
        plt.close() 