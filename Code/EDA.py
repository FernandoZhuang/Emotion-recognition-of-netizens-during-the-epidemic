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

    var_ = pd.read_csv('../Output/everyday_sentiment.csv', encoding='utf-8')
    col1, col2, col3 = var_['-1'].sum(), var_['0'].sum(), var_['1'].sum()

    print()
