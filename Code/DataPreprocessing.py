import my_setting.utils as utils
import pandas as pd
import os
from enum import IntEnum
import csv
import abc
from abc import ABCMeta
import re
from multiprocessing import Pool
from functools import partial, reduce
import time
import chinese_converter as cc
from collections import Counter


class DatasetType(IntEnum):
    LABELED: int = 0
    UNLABELED: int = 1
    TEST: int = 2
    SENTIMENTRELEVENTCORPUS: int = 3


class DataCleaningStep(metaclass=ABCMeta):

    def __init__(self, *args):
        start_time = time.time()
        self.run(args[0])
        end_time = time.time()
        # Unlabeled数据集不应该执行且输出LabelCheck信息
        if (len(args[0].columns) == 7 or args[1] != 'LabelCheck'):
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

    def _unqualified_label(self, x):
        '''
        :param x:[index, series]
        :return: 不合格index, [index]
        '''
        res = []
        for index, row in x:
            if ((row[6] == '0') | (row[6] == '1') | (row[6] == '-1') | (row[6] == 0) | (row[6] == 1) | (row[6] == -1)):
                continue
            else:
                res.append(index)

        return res


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
        # self._tools = [tool for tool in dir(self) if
        #                not tool.startswith('__') and not tool.endswith('__') and tool[0].isupper()]
        # 显式排序，配合Dataset中tool_whether_do的值
        self._tools = ['LabelCheck', 'DropExtraContentInTheEnd', 'DropHashtagAndAtReply',
                       'TraditionalChineseToSimplifiedChinese']

    def __getitem__(self, item):
        return getattr(self, item)

    class LabelCheck(DataCleaningStep):
        '''
        Label有各种噪音，暂时舍弃
        '''

        # TODO 等待讨论噪音标签的处理方法
        def run(self, dataset: pd.DataFrame, n_cores=10):
            # 如果是test，则直接pass
            if len(dataset.columns) == 7:
                # 非比赛数据，即自己搜集的数据label被解析成int，因此加入int类型判断
                # TODO 查找为什么会被解析成int
                print('---开始清洗Label noise---')
                with Pool(n_cores) as p:
                    res = p.map(self._unqualified_label, Batch(n_cores, list(dataset.iterrows())))
                    res = reduce(lambda x, y: x + y, res)

                dataset.drop(res, inplace=True)
                dataset.set_index(['id'], inplace=True)
                dataset.sentiment = dataset.sentiment.astype(int)
                # 分类训练时，n_class >=0 & n_class <= max_classes
                # 因此把-1映射到0,0映射到1,1映射到2
                dataset.sentiment = dataset.sentiment + 1

    class DropExtraContentInTheEnd(DataCleaningStep):
        """
        去除微博内容最后的"?"和"展开全文c"
        """

        def run(self, dataset: pd.DataFrame, n_cores=10):
            with Pool(n_cores) as p:
                exp = '\?展开全文c$|\?$|O网页链接'
                res = p.map(partial(self._regexp_sub, exp), Batch(n_cores, dataset.content.to_list()))
                res = reduce(lambda x, y: x + y, res)
                dataset.drop('content', inplace=True, axis=1)
                dataset.insert(2, 'content', res)

    class DropHashtagAndAtReply(DataCleaningStep):
        """
        去除"@账号名称"与"#hashtag"中的内容
        """

        def run(self, dataset: pd.DataFrame, n_cores=8):
            with Pool(n_cores) as p:
                exp = r'(#.*#)|(//@.*:)|(【.*】)'
                res = p.map(partial(self._regexp_sub, exp), Batch(n_cores, dataset.content.to_list()))
                res = reduce(lambda x, y: x + y, res)
                dataset.drop('content', inplace=True, axis=1)
                dataset.insert(2, 'content', res)

    class TraditionalChineseToSimplifiedChinese(DataCleaningStep):
        """
        繁体中文转为简体中文
        """

        def run(self, dataset: pd.DataFrame, n_cores: int = 8):
            with Pool(n_cores) as p:
                res = p.map(self._to_simplified, Batch(n_cores, dataset.content.to_list()))
                res = reduce(lambda x, y: x + y, res)
                dataset.drop('content', inplace=True, axis=1)
                dataset.insert(2, 'content', res)

    @property
    def tools(self):
        return self._tools


class Dataset(pd.DataFrame):

    def __init__(self, path: str, type_: int, tool_whether_do=4):
        """
        数据文件类初始化函数，直接继承自DataFrame
        :param path: 数据文件路径
        :param type_: 数据文件类型
        :param tool_whether_do: 数据清洗执行到哪步
        依次是'LabelCheck', 'DropExtraContentInTheEnd', 'DropHashtagAndAtReply',
        'TraditionalChineseToSimplifiedChinese'
        """

        # 以下为读取数据部分
        pd.DataFrame.__init__(self)
        assert os.path.exists(path), '数据文件路径错误！'

        if type_ == DatasetType.SENTIMENTRELEVENTCORPUS:
            self._data = pd.read_csv(path, usecols=[1, 2, 3, 4, 5], )._data
            print('已读入标注数据集')
            self.index.name = 'ID'
        else:
            dateparser = lambda x: pd.datetime.strptime(x, '%m月%d日 %H:%M')
            if type_ == DatasetType.LABELED:
                self._data = pd.read_csv(path, usecols=[0, 1, 2, 3, 4, 5, 6], parse_dates=['微博发布时间'],
                                         date_parser=dateparser)._data
                print('已读入标注数据集')
                self.columns = ['id', 'datetime', 'poster', 'content', 'image', 'video', 'sentiment']
                # self.index.name = 'ID'
            else:
                # 若要区分train unlabel test，则可基于本else代码段重新改写
                self._data = pd.read_csv(path, index_col=['微博id'], parse_dates=['微博发布时间'],
                                         date_parser=dateparser)._data
                print('已读入无标注数据集')
                self.columns = ['datetime', 'poster', 'content', 'image', 'video']
                self.index.name = 'ID'
                self.datetime += pd.Timedelta(120 * 365, unit='d')

        self._cleaned_data = None
        self._cat_hashtags = None

        # 以下为注册要执行的数据清理工具部分
        self.tool_set = DataCleaningToolSet()
        self.registered_tools = []
        self.register_data_clean_tools(self.tool_set.tools, tool_whether_do)

    def register_data_clean_tools(self, tools: list, flag: int):
        cnt = 1

        for tool in tools:
            assert tool in self.tool_set.tools, f"清洗工具{tool}不存在"
            if cnt > flag: break
            self.registered_tools.append(tool)
            cnt += 1

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
                self.tool_set[tool](self._cleaned_data, tool)
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
            print('正在统计hashtag...', sep='')
            p = Pool(processes=n_core)
            res = p.map(self._find_hashtags, Batch(n_core, self.content.to_list()))
            res = p.map(self._list_reduce, Batch(3, res))
            res = reduce(lambda x, y: x + y, res)
            self._cat_hashtags = Counter(res)
            print('完毕')
            return self._cat_hashtags


class LabeledDataset(Dataset):

    def __init__(self, flag: int = 1, path: str = utils.cfg.get('ORIGINAL_DATA', 'train_labeled_path')):
        Dataset.__init__(self, path, DatasetType.LABELED, flag)


class UnlabeledDataset(Dataset):

    def __init__(self, flag: int = 1, path: str = utils.cfg.get('ORIGINAL_DATA', 'train_unlabeled_path')):
        Dataset.__init__(self, path, DatasetType.UNLABELED, flag)


class TestDataset(Dataset):

    def __init__(self, flag: int = 1, path: str = utils.cfg.get('ORIGINAL_DATA', 'test_path')):
        Dataset.__init__(self, path, DatasetType.TEST, flag)

    def submit(self, path: str = utils.cfg.get('PROCESSED_DATA', 'submit_csv_path')):
        """
        生成排行榜提交文件
        :param path: 排行榜文件的输出路径
        :return:
        """
        with open(path, 'w') as f:
            writer = csv.DictWriter(f, ['id', 'y'])
            writer.writeheader()
            for idx, row in self.iterrows():
                item = {'id': str(idx), 'y': str(row['sentiment'] - 1)}  # -1 是因为预测时标签为自然数，而提交结果却是-1,0,1
                writer.writerow(item)

    def fill_result(self, res: list):
        """
        填充预测结果
        :param res: 结果
        :return:
        """
        if 'sentiment' in self.columns:
            self.drop('sentiment', inplace=True)
        # 原始self.insert(-1,,,)报错unbounded slice
        # http://sofasofa.io/forum_main_post.php?postid=1003010
        self.insert(self.shape[1], 'sentiment', res)


def sentiment_relevent_corpus():
    '''
    处理情感分析领域相关语料
    :return:
    '''
    # https://zhuanlan.zhihu.com/p/80029681
    # region weibo_senti_100k数据集
    weibo_senti_100k = 1
    if weibo_senti_100k == 1:
        senti = pd.read_csv(
            '/home/zhw/PycharmProjects/nCovSentimentAnalysis/Data/SentimentRelevantCorpus/unzip/chineseNIP_weibo_senti_100k.csv',
            encoding='utf-8')
        columns_titles = ['review', 'label']
        senti = senti.reindex(columns=columns_titles)
        # 比赛数据 label是str类型，而不是nt类型
        senti['label'] = senti['label'].apply(lambda x: '-1' if x == 0 else '1')
        senti.columns = ['content', 'sentiment']
        # 插空列，保持和比赛数据格式一致
        senti = senti.reindex(columns=['datetime', 'poster', 'content', 'image', 'video', 'sentiment'])
        senti.to_csv('relevent_senti_100k.csv', index=False)
    # endregion

    # region simplifyweibo_4_moods
    senti = pd.read_csv(
        '/home/zhw/PycharmProjects/nCovSentimentAnalysis/Data/SentimentRelevantCorpus/unzip/simplifyweibo_4_moods.csv',
        encoding='utf-8')

    columns_titles = ['review', 'label']
    senti = senti.reindex(columns=columns_titles)

    senti['label'] = senti['label'].apply(lambda x: '-1' if x != 0 else '1')
    senti.columns = ['content', 'sentiment']
    senti = senti.reindex(columns=['datetime', 'poster', 'content', 'image', 'video', 'sentiment'])
    senti.to_csv('simplify_weibo_360k.csv', index=False)
    # endregion


def sample_add_sentiment():
    '''
    数据的输入是用模型打好了伪标签的900k csv（testdataset.submit函数生成）和原始900k csv
    目的是按一定比例抽样，获取900k csv中的一部分映射了标签的数据，送入模型和100k结合再训练
    参考https://stackoverflow.com/questions/37047420/how-to-drop-rows-of-pandas-dataframe-with-same-value-based-on-condition-in-diffe
    '''
    # TODO 由于UnlabelDataset初始化得先读入900k等一系列操作，耗时大，暂时不放在UnlabelDataset，等待未来优化
    # TODO 随机采样，有一定概率在train_unlabel_sample.insert报错Length of values does not match length of index
    # 暂时解决方案：重试。等待完善删除所有重复
    assert os.path.exists(utils.cfg.get('PROCESSED_DATA', 'unlabel_pseudo_path')), 'unlabel_pseudo文件路径错误或不存在或命名错误！'
    sentiment_polar = pd.read_csv(utils.cfg.get('PROCESSED_DATA', 'unlabel_pseudo_path'), encoding='utf-8')
    train_unlabel = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_unlabel_path'), encoding='utf-8')
    train_unlabel.columns = ['ID', 'datetime', 'poster', 'content', 'image', 'video']

    train_unlabel.insert(loc=6, column='sentiment', value=sentiment_polar['y'].to_list())
    train_unlabel_sample = train_unlabel.sample(frac=0.1)  # frac是抽样比例
    train_unlabel_sample.to_csv(utils.cfg.get('PROCESSED_DATA', 'unlabel_sample_path'), index=False)

    # 合并文件
    li = []
    label = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_labeled_path'), encoding='utf-8', header=0)
    label.columns = ['ID', 'datetime', 'poster', 'content', 'image', 'video', 'sentiment']
    li.append(label), li.append(train_unlabel_sample)
    mix_lable_unlabel = pd.concat(li, axis=0, ignore_index=True)
    # 适配Dataset label 判断里index_col='微博id'
    mix_lable_unlabel.columns = ['微博id', 'datetime', 'poster', 'content', 'image', 'video', 'sentiment']
    mix_lable_unlabel.to_csv(utils.cfg.get('PROCESSED_DATA', 'mix_label_unlabel_path'), index=False)

# if __name__ == '__main__':
#     # testset = LabeledDataset()
#     # print(testset.cleaned_data)
#
#     sample_add_sentiment()
