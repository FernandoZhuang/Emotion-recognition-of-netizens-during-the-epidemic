import pandas as pd
import numpy as np
import os
from enum import IntEnum
import csv
import abc
from abc import ABCMeta
import re
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial, reduce
import time
import chinese_converter as cc
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class DatasetType(IntEnum):
    LABELED: int = 0
    UNLABELED: int = 1
    TEST: int = 2


class DataCleaningStep(metaclass=ABCMeta):

    def __init__(self, *args):
        start_time = time.time()
        self.run(args[0])
        end_time = time.time()
        print(f'已执行数据清洗步骤：{self.__doc__.strip()}，用时：{round(end_time - start_time, 2)}s')

    @abc.abstractmethod
    def run(self, dataset: pd.DataFrame, n_cores: int = 8):
        pass

    def _regexp_sub(self, regexp: str, x: list, substr: str = ''):
        """
        清除满足根据正则表达式的列表x中的内容
        :param regexp: 这则表达式
        :param x: 内容列表
        :return: 清理后的内容列表
        """
        data = []
        for content in x:
            new_content = re.sub(regexp, substr, str(content), flags=re.U)
            data.append(new_content)
        return data

    def _regexp_findall(self, regexp, x: list):
        """
        返回满足正则表达式的字符
        :param regexp:
        :param x:
        :return:
        """
        return [re.findall(regexp, pattern) for pattern in x]

    def _to_simplified(self, x):
        return [cc.to_simplified(proc) for proc in x]


class Batch:
    """
    数据分块，用户多进程加速
    """

    def __init__(self, n_group, data):
        """
        构造函数
        :param n_group: 待分组数
        :param data: 待分数据集
        """
        self._n_group = n_group
        self._data = data
        n_data = len(self._data)
        groups = []
        if int(n_data / self._n_group) == 0:
            groups.append((0, n_data))
        else:
            b = int(n_data / self._n_group)
            for i in range(self._n_group):
                if i != self._n_group - 1:
                    groups.append((i * b, i * b + b))
                else:
                    groups.append((i * b, n_data))
        self._groups = groups

    def __iter__(self):
        for s in self._groups:
            yield self._data[s[0]: s[1]]


class DataCleaningToolSet:
    """
    数据清理步骤的集合
    """

    def __init__(self):
        self._tools = [tool for tool in dir(self) if
                       not tool.startswith('__') and not tool.endswith('__') and tool[0].isupper()]

    def __getitem__(self, item):
        return getattr(self, item)

    class DropExtraContentInTheEnd(DataCleaningStep):
        """
        去除微博内容最后的"?"和"展开全文c"
        """

        def run(self, dataset: pd.DataFrame, n_cores=6):
            p = Pool(n_cores)
            exp = '\?展开全文c$|\?$|O网页链接'
            res = p.map(partial(self._regexp_sub, exp), Batch(n_cores, dataset.content.to_list()))
            res = reduce(lambda x, y: x + y, res)
            dataset.drop('content', inplace=True, axis=1)
            dataset.insert(2, 'content', res)
            p.close()

    class DropHashtagAndAtReply(DataCleaningStep):
        """
        去除"@账号名称"与"#hashtag"中的内容
        """

        def run(self, dataset: pd.DataFrame, n_cores=8):
            p = Pool(n_cores)
            exp = r'(#.*#)|(//@.*:)|(【.*】)'
            res = p.map(partial(self._regexp_sub, exp), Batch(n_cores, dataset.content.to_list()))
            res = reduce(lambda x, y: x + y, res)
            dataset.drop('content', inplace=True, axis=1)
            dataset.insert(2, 'content', res)
            p.close()

    class TraditionalChineseToSimplifiedChinese(DataCleaningStep):
        """
        繁体中文转为简体中文
        """

        def run(self, dataset: pd.DataFrame, n_cores: int = 8):
            p = Pool(n_cores)
            res = p.map(self._to_simplified, Batch(n_cores, dataset.content.to_list()))
            res = reduce(lambda x, y: x + y, res)
            dataset.drop('content', inplace=True, axis=1)
            dataset.insert(2, 'content', res)
            p.close()

    class LabelCheck(DataCleaningStep):
        """
        去除标签噪声
        """
        def run(self, dataset: pd.DataFrame, n_cores: int = 8):
            cleaned_data = dataset[(dataset['sentiment'] == '0') | (dataset['sentiment'] == '-1') | (dataset['sentiment'] == '1')]
            cleaned_data.sentiment = cleaned_data.sentiment.astype('int')
            dataset._data = cleaned_data._data


    @property
    def tools(self):
        return self._tools


class Dataset(pd.DataFrame):

    def __init__(self, path: str, type_: int):
        """
        数据文件类初始化函数，直接继承自DataFrame
        :param path: 数据文件路径
        :param type_: 数据文件类型
        """

        # 以下为读取数据部分
        pd.DataFrame.__init__(self)
        assert os.path.exists(path), '数据文件路径错误！'
        if type_ == DatasetType.LABELED:
            self._data = pd.read_csv(path, index_col=['微博id'])._data
            print('已读入标注数据集')
            self.columns = ['poster', 'content', 'image', 'video', 'sentiment']
            self.index.name = 'ID'
        elif type_ == DatasetType.UNLABELED:
            dateparser = lambda x: pd.datetime.strptime(x, '%m月%d日 %H:%M')
            self._data = pd.read_csv(path, index_col=['微博id'], parse_dates=['微博发布时间'], date_parser=dateparser)._data
            print('已读入未标注数据集')
            self.columns = ['datetime', 'poster', 'content', 'image', 'video']
            self.index.name = 'ID'
            self.datetime += pd.Timedelta(120 * 365, unit='d')
        else:
            dateparser = lambda x: pd.datetime.strptime(x, '%m月%d日 %H:%M')
            self._data = pd.read_csv(path, index_col=['微博id'], parse_dates=['微博发布时间'], date_parser=dateparser)._data
            print('已读入测试集')
            self.columns = ['datetime', 'poster', 'content', 'image', 'video']
            self.index.name = 'ID'
            self.datetime += pd.Timedelta(120 * 365, unit='d')

        self._cleaned_data = None
        self._cat_hashtags = None
        self._emojis = None

        # 以下为注册要执行的数据清理工具部分
        self.tool_set = DataCleaningToolSet()
        self.registered_tools = []
        self.register_data_clean_tools(self.tool_set.tools)

    def register_data_clean_tools(self, tools: list):
        for tool in tools:
            assert tool in self.tool_set.tools, f"清洗工具{tool}不存在"
            self.registered_tools.append(tool)

    @property
    def cleaned_data(self):
        """
        获取清洗过的数据，原数据集中的内容不变。若数据没有清洗，则进行清洗；若数据清洗过，则直接返回清洗过的数据。
        :return:
        """
        if self._cleaned_data is not None:
            return self._cleaned_data
        else:
            self._cleaned_data = self.copy(deep=True)
            for tool in self.registered_tools:
                self.tool_set[tool](self._cleaned_data)
            return self._cleaned_data

    def _find_hashtags(self, x):
        return [re.findall('#(.+?)#', str(content)) for content in x]

    def _list_reduce(self, l):
        res = []
        for l1 in l:
            for l2 in l1:
                if len(l2) != 0:
                    res.extend(l2)
        return res

    @property
    def stat_hashtags(self, n_core=8):
        """
        获取数据集的hashtag
        :return:
        """

        p = Pool(n_core)
        if self._cat_hashtags is not None:
            return self._cat_hashtags
        else:
            print('正在统计hashtag...', end='')
            p = Pool(processes=n_core)
            res = p.map(self._find_hashtags, Batch(n_core, self.content.to_list()))
            res = p.map(self._list_reduce, Batch(3, res))
            res = reduce(lambda x, y: x + y, res)
            self._cat_hashtags = Counter(res)
            print('完毕')
            return self._cat_hashtags

    def _extract_emoji(self, x):
        ret = []
        for idx, row in x:
            res = re.findall(r'\[.+?\]', str(row))
            res = list(set(filter(lambda x:0 < len(x) <= 6, res)))
            if len(res) > 0:
                ret.append((idx, res))
        return ret


    @property
    def emojis(self):
        """
        提取emoji表情
        :return:
        """
        n_core = 8
        if self._emojis is not None:
            return self._emojis
        else:
            print('正在提取emoji...', end='')
            p = Pool(processes=n_core)
            res = p.map(self._extract_emoji, Batch(n_core, list(self.content.to_dict().items())))
            res = reduce(lambda x, y: x + y, res)
            self._emojis = res
            print('完毕')
            return self._emojis

class LabeledDataset(Dataset):

    def __init__(self, path: str = r'./data/nCoV_100k_train.labled.csv'):
        Dataset.__init__(self, path, DatasetType.LABELED)


class UnlabeledDataset(Dataset):

    def __init__(self, path: str = r'./data/nCoV_900k_train.unlabled.csv'):
        Dataset.__init__(self, path, DatasetType.UNLABELED)


class TestDataset(Dataset):

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
                item = {'id': str(idx), 'y': str(row['sentiment'])}
                writer.writerow(item)

    def fill_result(self, res: list):
        """
        填充预测结果
        :param res: 结果
        :return:
        """
        if 'sentiment' in self.columns:
            self.drop('sentiment', inplace=True)
        self.insert(-1, 'sentiment', res)


if __name__ == '__main__':
    testset = LabeledDataset()
    print(testset.cleaned_data.sentiment)
