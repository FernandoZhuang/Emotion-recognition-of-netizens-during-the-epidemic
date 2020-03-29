import pandas as pd
import cufflinks as cf
import jieba
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import seaborn as sns

from wordcloud import WordCloud
import my_setting.utils as utils

if __name__ == '__main__':
    cf.go_offline()
    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 100)
    plt.style.use('seaborn')

    sentiment_polar = pd.read_csv('/home/zhw/PycharmProjects/nCovSentimentAnalysis/Output/submit_file.csv',
                                  encoding='utf-8')
    train_unlabel = pd.read_csv(utils.cfg.get('ORIGINAL_DATA', 'train_unlabeled_path'), encoding='utf-8')
    train_unlabel.columns = ['ID', 'datetime', 'poster', 'content', 'image', 'video']

    sentiment_sample = sentiment_polar.sample(frac=0.1)
    sentiment_sample.sort_values('id', inplace=True)
    train_unlabel_sample = train_unlabel.loc[train_unlabel['ID'].isin(sentiment_sample['id'].to_list())]
    train_unlabel_sample.sort_values('ID', inplace=True)

    train_unlabel_sample.drop_duplicates(subset=['ID'], keep='first', inplace=True)

    # duplicated_row=train_unlabel_sample.loc[train_unlabel_sample['ID'].duplicated(keep='first'),'ID']
    # unlabel_isin=train_unlabel_sample['ID'].isin(duplicated_row.unique())
    # unlabel_index=train_unlabel_sample.index[unlabel_isin]
    # train_unlabel_sample.drop(unlabel_index, inplace=True)

    train_unlabel_sample.insert(loc=6, column='sentiment', value=sentiment_sample['y'].to_list())
    train_unlabel_sample.to_csv('unlabel_sample.csv', index=False)

    print()
