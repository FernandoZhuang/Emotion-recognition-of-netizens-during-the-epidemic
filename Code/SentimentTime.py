import re
import pandas as pd
import multiprocessing
import functools
import time

import my_setting.utils as utils
import DataPreprocessing as dp


class SentimentTime():
    def __init__(self, train_label=True, train_unlabel=False, test=False):
        if train_label: self.train_label = dp.LabeledDataset(1).cleaned_data
        if train_unlabel: self.train_unlabel = dp.UnlabeledDataset(1).cleaned_data
        if test: self.test = dp.TestDataset(1).cleaned_data

    def _calculate(self, x):
        '''
        :param x: list[iterrows]
        :return:
        '''
        res = pd.DataFrame(columns=['dayfromzero', '-1', '0', '1'])
        res.set_index(['dayfromzero'], inplace=True)

        oneday, startday = [0, 0, 0], x[0][1][8]
        for _, row in x:
            if row[8] != startday:
                res.loc[startday] = oneday
                oneday, startday = [0, 0, 0], row[8]
            oneday[int(row[5])] += 1

        return res

    def everyday_sentiment(self):
        '''
        计算情感极性在每天的概率分布
        :return:
        '''
        self.train_label['month'] = self.train_label['datetime'].dt.month
        self.train_label['day'] = self.train_label['datetime'].dt.day
        self.train_label['dayfromzero'] = (self.train_label['month'] - 1) * 31 + self.train_label['day']

        start_time = time.time()
        print('---情感极性在每天概率分布开始计算---')
        with multiprocessing.Pool(10) as p:
            res = p.map(self._calculate, dp.Batch(10, list(self.train_label.iterrows())))
            res = functools.reduce(lambda x, y: x + y, res)

        print(f'情感极性在每天概率分布计算完毕，耗时{time.time() - start_time}s')
        res.to_csv('everyday_sentiment.csv', header=False)
        return res

    def window_time_sentiment(self):
        '''
        计算情感极性在一个window size 时间尺寸内的概率分布
        :return:
        '''

        print()


if __name__ == '__main__':
    worker = SentimentTime()
    worker.everyday_sentiment()

    print()
