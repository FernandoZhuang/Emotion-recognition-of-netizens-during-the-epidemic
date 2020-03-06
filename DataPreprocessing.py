import pandas as pd
import numpy as np
import os
from enum import IntEnum
import csv


class DatasetType(IntEnum):
    LABELED: int = 0
    UNLABELED: int = 1
    TEST: int = 2


class Dataset(pd.DataFrame):

    def __init__(self, path: str, type_: int):
        """
        数据文件类初始化函数，直接继承自DataFrame
        :param path: 数据文件路径
        :param type_: 数据文件类型
        """
        pd.DataFrame.__init__(self)
        assert os.path.exists(path), '数据文件路径错误！'
        if type_ == DatasetType.LABELED:
            self._data = pd.read_csv(path, index_col=['微博id'])._data
            self.columns = ['poster', 'content', 'image', 'video', 'sentiment']
            self.index.name = 'ID'
        elif type_ == DatasetType.UNLABELED:
            dateparser = lambda x: pd.datetime.strptime(x, '%m月%d日 %H:%M')
            self._data = pd.read_csv(path, index_col=['微博id'], parse_dates=['微博发布时间'], date_parser=dateparser)._data
            self.columns = ['datetime', 'poster', 'content', 'image', 'video']
            self.index.name = 'ID'
            self.datetime += pd.Timedelta(120, unit='y')
        else:
            dateparser = lambda x: pd.datetime.strptime(x, '%m月%d日 %H:%M')
            self._data = pd.read_csv(path, index_col=['微博id'], parse_dates=['微博发布时间'], date_parser=dateparser)._data
            self.columns = ['datetime', 'poster', 'content', 'image', 'video']
            self['sentiment'] = None
            self.index.name = 'ID'
            self.datetime += pd.Timedelta(120, unit='y')


class LabeledDataset(Dataset):

    def __init__(self, path: str = r'./data/nCoV_100k_train.labled.csv'):
        Dataset.__init__(self, path, DatasetType.LABELED)


class UnlabeledDataset(Dataset):

    def __init__(self, path: str = r'./data/nCoV_900k_train.unlabled.csv'):
        Dataset.__init__(self, path, DatasetType.UNLABELED)


class Testset(Dataset):

    def __init__(self, path: str = r'./data/nCoV_10k_test.csv'):
        Dataset.__init__(self, path, DatasetType.TEST)

    def submit(self, path: str = r'./submit_file.csv'):
        """
        生成排行榜提交文件
        :param path: 排行榜文件的输出路径
        :return:
        """
        with open(path, 'w') as f:
            writer = csv.DictWriter(f, ['id', 'y'])
            writer.writeheader()
            for idx, row in self.iterrows():
                item = {'id': str(idx) + ' ', 'y': str(row['sentiment'])}
                writer.writerow(item)


if __name__ == '__main__':
    pass
