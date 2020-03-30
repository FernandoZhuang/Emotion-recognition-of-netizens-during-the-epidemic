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

    train = pd.read_csv(
        './knn-classification-master/knn-classification/data/cnews.train.knn_seg',
        encoding='utf-8')

    print()
