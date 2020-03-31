import pandas as pd
import numpy as np
import re
import multiprocessing
import functools
import time
import collections
import tqdm

import my_setting.utils as utils
import DataPreprocessing as dp
import Emoj


class Hashtag():
    def __init__(self, train_label=True, train_unlabel=False, test=True):
        if train_label:
            self.train_label = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_labeled_path'), encoding='utf-8')
        if train_unlabel:
            self.train_unlabel = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_unlabeled_path'), encoding='utf-8')
        if test:
            self.test = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'test_path'), encoding='utf-8')
        # TODO 计划根据传递的动态化初始参数,例如train_label,train_unlabel,test都为True，支持同时返回对应hashtag，则content因为list
        self.content = self.train_label['微博中文内容'].to_list()
        self.label = self.train_label['情感倾向'].to_list()

        self.exp = r'#(.+?)#'

    def get_hashtag_concat_weibo_content(self, flag=0):
        '''
        #TODO 把数据类型转换成dataframe,尝试加快速度，优化代码结构
        flag:应用场景 一条微博里里有多种hashtag（多个但相同hashtag，不重复映射）
        0 只保存第一次匹配到的hahstag
        1 重复多行，即把多种hashtag映射到同一句话，行的起始只有hashtag不同，微博内容相同
        :return:
        返回参数用于KNN输入
        格式 [hashtag weibo_content]
        '''
        res = []

        for row in self.content:
            hashtag = set([tmp for tmp in re.findall(self.exp, str(row))])
            if hashtag:
                for item in hashtag:
                    tmp = item + '\t' + row
                    res.append(tmp)
                    if flag == 0: break

        return res

    def get_content_without_hashtag(self):
        '''
        :return:
        获取微博，该类微博里没有hashtag
        '''
        res = []

        for row in self.content:
            hashtag = re.match(self.exp, str(row))
            if hashtag is None: res.append(row)

        return res

    def get_hashtag(self, x: list):
        '''
        :return:
        1. train_label 有16504种不同标签
        '''
        return set([tmp for row in x for tmp in re.findall(self.exp, str(row))])

    def distribution(self):
        '''
        计算hashtag在某时间点，某日的情感极性概率分布
        作为bayes的先验知识
        :return:
        '''
        # train100K 平均每种hashtag有3.4条微博, train900K平均每种有8条，test则为1.7222条
        thresh = [4, 8, 2]

        return self._whole_distribution(thresh, False)

    def sentiment_distribution_of_one_hashtag(self, hashtags: list):
        '''
        计算一种hashtag对应的各个极性分布，和并行结合
        :return:
        '''
        # TODO 等待兼容train 900K. if is is not None需要着重更改，达到优化
        # TODO 等待优化dataframe的插入
        res = pd.DataFrame(columns=['hashtag', '-1', '0', '1'])
        res.set_index(['hashtag'], inplace=True)
        record = set()  # 判断hashtag是否已存在于dataFrame中，处理key error

        for row in self.train_label.iterrows():
            tmps = set([tag for tag in re.findall(self.exp, str(row[1][3]))])
            for tmp in tmps:
                if tmp in record:
                    if row[1][6] == '-1':
                        res.loc[tmp][0] += 1
                    elif row[1][6] == '0':
                        res.loc[tmp][1] += 1
                    else:
                        res.loc[tmp][2] += 1
                else:
                    li = [0, 0, 0]
                    if row[1][6] == '-1':
                        li[0] += 1
                    elif row[1][6] == '0':
                        li[1] += 1
                    else:
                        li[2] += 1
                    res.loc[tmp] = li
                    record.add(tmp)

        return res

    def _whole_distribution(self, thresh: list, thresh_flag=False):
        '''
        计算各个Hashtag极性在总时间内的概率分布
        :return:
        '''
        start_time = time.time()

        if thresh_flag is False:
            n_cores = 10
            print('---获取hashtag极性在总时间内的概率分布---')
            with multiprocessing.Pool(n_cores) as p:
                res = p.map(self.sentiment_distribution_of_one_hashtag,
                            dp.Batch(n_cores, list(self.get_hashtag(self.content))))
                res = functools.reduce(lambda x, y: x + y, res)

            for row in res.iterrows():
                # HACK 放入多进程中，会有大量大于1的数，所以单独拿出来
                sum = row[1][0] + row[1][1] + row[1][2]
                row[1][0], row[1][1], row[1][2] = float(row[1][0]) / sum, float(row[1][1]) / sum, float(row[1][2]) / sum

            res.to_csv('distribution_all_hashtag.csv', header=False)
            print(f'---获取用时:{round(time.time() - start_time, 2)}s---')

        return res

    def _day_by_day_distribution(self, thresh: list, thresh_flag=False):
        '''
        计算各个Hashtag极性分别在每天的概率分布
        :return:
        '''

    def bayes(self, logits: list):
        '''
        logits: [narray,narray,...]
        利用先验知识，修正情感极性输出
        :return: [narray, narray, ....]
        '''
        # TODO 等待完成纠正train部分，做好兼容
        # TODO 一句话里有多个hashtag兼容
        cnt = 0
        res = []
        test_content = self.test['微博中文内容'].to_list()
        # 使用train部分的hashtag,目前阶段只计算了train100K的hashtag极性分布
        hash_tag = self.get_hashtag(self.content)
        distribution = self.distribution()

        for batch_logit in logits:
            li = []
            for logit in batch_logit:
                tmp = re.findall(self.exp, str(test_content[cnt]))
                if len(tmp) == 1 and (tmp[0] in hash_tag):
                    li += [[distribution.loc[tmp[0]][i] * logit[i] for i in range(3)]]
                else:
                    li += [logit]
                cnt += 1

            res += [np.asarray(li)]

        return res


class TextCluster():
    def __init__(self):
        '''
        把无hashtag微博聚类到已有的hashtag
        结合emoj
        '''


if __name__ == '__main__':
    worker = Hashtag()

    res = worker.get_content_without_hashtag()
    worker.distribution()

    print()
