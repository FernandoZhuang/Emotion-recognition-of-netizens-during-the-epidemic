import math
import os
import pandas as pd
import numpy as np
import time

import my_setting.utils as utils
import DataPreprocessing as dp


class SentimentTime():
    def __init__(self, train_label=False, train_unlabel=False, test=False):
        if train_label: self.train_label = dp.LabeledDataset(1).cleaned_data
        if train_unlabel: self.train_unlabel = dp.UnlabeledDataset(1).cleaned_data
        if test: self.test = dp.TestDataset(1).cleaned_data

    def everyday_sentiment(self):
        '''
        计算情感极性在每天的概率分布
        :return:
        '''
        if os.path.exists(utils.cfg.get('PROCESSED_DATA', 'everyday_sentiment_path')):
            return pd.read_csv(utils.cfg.get('PROCESSED_DATA', 'everyday_sentiment_path'))

        self.train_label['month'] = self.train_label['datetime'].dt.month
        self.train_label['day'] = self.train_label['datetime'].dt.day
        self.train_label['dayfromzero'] = (self.train_label['month'] - 1) * 31 + self.train_label['day']

        start_time = time.time()
        print('---情感极性在每天概率分布开始计算---')
        res = pd.DataFrame(columns=['dayfromzero', '-1', '0', '1'])
        res.set_index(['dayfromzero'], inplace=True)

        day_flag = set()
        for _, row in self.train_label.iterrows():
            if row[8] not in day_flag:
                li = [0, 0, 0]
                li[row[5]] += 1
                res.loc[row[8]] = li
                day_flag.add(row[8])
            else:
                res.loc[row[8]][row[5]] += 1

        # res.to_csv(utils.cfg.get('PROCESSED_DATA', 'everyday_sentiment_path'))
        return res

    def window_time_sentiment(self, window: int = 5):
        '''
        计算情感极性在一个window size 时间尺寸内的概率分布
        :return:
        '''
        res = pd.DataFrame(columns=['window', '-1', '0', '1'])
        res.set_index(['window'], inplace=True)
        eachday = self.everyday_sentiment()

        window_day, window_set = [0, 0, 0], set()
        for _, row in eachday.iterrows():
            var_ = math.floor(row[0] / window)
            if var_ not in window_set:
                res.loc[var_] = [row[1], row[2], row[3]]
                window_set.add(var_)
            else:
                res.loc[var_] = [res.loc[var_][i] + row[i + 1] for i in range(3)]

        for index, row in res.iterrows():
            sum = row[0] + row[1] + row[2]
            res.loc[index] = [float(row[i] / sum) for i in range(3)]

        res.to_csv(utils.cfg.get('PROCESSED_DATA', 'window_time_sentiment_path'))
        return res

    def bayes(self, logits: list, window: int = 1):
        '''
        依据情感极性随时间变化纠正神经网络输出
        :param window代表使用多长时间范围纠正
        :return:
        '''
        cnt, res = 0, []
        self.test['month'] = self.test['datetime'].dt.month
        self.test['day'] = self.test['datetime'].dt.day
        self.test['dayfromzero'] = (self.test['month'] - 1) * 31 + self.test['day']
        window_record = self.window_time_sentiment(window)

        print('---开始按照情感极性随时间纠正---')
        for batch_logit in logits:
            li = []
            for logit in batch_logit:
                index = math.floor(self.test.iat[cnt, 7] / window)
                li += [[window_record.loc[index][i] * logit[i] for i in range(3)]]
                cnt += 1

            res += [np.asarray(li)]

        return res

#
# if __name__ == '__main__':
#     worker = SentimentTime(test=True)
#     worker.bayes(list())
#
#     print()
