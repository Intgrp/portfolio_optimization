#!/usr/bin/python
# -*- coding:utf-8 -*-
import os.path

import numpy as np
import pandas as pd
from typing import List, Tuple
from .data_loader import DataLoader


class CsvDataLoader(DataLoader):
    """文本拉取生成器"""

    def __init__(self, data_path:str, assets: List[str], freq: str="D", category: str="品种净收益率"):
        """
        初始化文件信息
        data/D/xxx.csv 日线每日行情
        data/品种板块周期的净收益率.xlsx

        Parameters
        ----------
        data_path : str 例如 data/
            文件列表
        """
        self.data_path = data_path
        self.freq = freq
        self.assets = assets
        # 收益率表选取的是品种的还是板块还是周期
        self.category = category


    def load_price_df(self, ins, usecols=None):
        file_path = os.path.join(self.data_path, self.freq, f'{ins}.csv')
        if not os.path.exists(file_path):
            raise FileExistsError(f"路径文件不存在：{file_path}")
        df = pd.read_csv(file_path, usecols=usecols)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df

    def load_ins_list_close_df(self):
        df_result = pd.DataFrame()
        for ins in self.assets:
            df_temp = self.load_price_df(ins, usecols=['datetime', 'close'])
            df_result[ins] = df_temp['close']
        return df_result


    def get_category_returns(self):
        file_name = os.path.join(self.data_path, '品种板块周期的净收益率.xlsx')
        df_data = pd.read_excel(file_name, sheet_name=self.category)
        df_data['时间'] = pd.to_datetime(df_data['时间'])
        df_data.set_index('时间', inplace=True)
        return df_data

    def load_data(self, start_date: str, end_date: str):
        returns = self.get_category_returns()
        prices = self.load_ins_list_close_df()
        return prices.loc[start_date: end_date, :], returns.loc[start_date: end_date, :]

