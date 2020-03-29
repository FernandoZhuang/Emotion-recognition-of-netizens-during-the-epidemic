import pandas as pd
import re

import my_setting.utils as utils
import Emoj


class Hashtag():
    def __init__(self):
        print()

    def _get_hashtag(self):
        '''
        获取Hashtag
        :return:
        '''
        train_label = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_labeled_path'), encoding='utf-8')
        train_unlabel = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_unlabeled_path'), encoding='utf-8')
        test = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'test_path'), encoding='utf-8')

        exp = r'#(.+?)#'
        content = train_label['微博中文内容'].to_list()

        return self._regex(exp, content)

    def _regex(self, exp: str, x: list):
        '''
        :return:
        1. train_label 有16504种不同标签
        '''
        return set([tmp for row in x for tmp in re.findall(exp, str(row))])

    def _distribution(self):
        '''
        计算hashtag在某时间点，某日的情感极性概率分布
        作为bayes的先验知识
        :return:
        '''

    def bayes(self):
        '''
        利用先验知识，修正test情感极性输出
        :return:
        '''


class Cluster():
    def __init__(self):
        '''
        把无hashtag微博聚类到已有的hashtag
        结合emoj
        '''

    def _KNN(self):
        '''
        选取KNN聚类
        :return:
        '''


if __name__ == '__main__':
    worker = Hashtag()

    res = worker._get_hashtag()

    print()
