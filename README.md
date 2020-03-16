# 数据预处理
## I/O
2020-03-06 
1. 新增三个数据类可以快速将数据读入DataFrame中, 并格式化日期 \
这三个数据类为：LabeledDataset、UnlabeledDataset、Testset \
将中文标题重命名为：
    - 微博id -> ID(主键)
    - 微博发布时间 -> datetime
    - 发布人账号 -> poster
    - 微博中文内容 -> content
    - 微博图片 -> image
    - 微博视频 -> video
    - 情感倾向 -> sentiment
2. 新增生成测试数据集的排行榜提交文件函数submit，实现了ID号后需加空格的坑人逻辑
2020-03-07
3. 测试集TestDataset新增fill_result函数，可以用来填充模型训练的结果
## 数据清洗
2020-03-07
1. 访问数据集属性cleaned_data可以执行多进程加速的清洗步骤并获得清洗过的数据集，若cleaned_data是第二次访问，则无需执行清洗步骤，已实现的清洗步骤为：
    - 去除微博末尾的无意义的文字
    - 去除@回复中的微博名称及hashtag、【】中的内容
    - 繁体中文转简体中文
## 统计
1. 属性stat_hashtags用来提取不与微博关联的所有hashtag与该hashtag出现的次数


## 参考文献
1. https://arxiv.org/pdf/1908.10084.pdf, Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
2. https://github.com/UKPLab/sentence-transformers
3. https://medium.com/genei-technology/richer-sentence-embeddings-using-sentence-bert-part-i-ce1d9e0b1343
4. https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
5. https://github.com/andrewgordonwilson/bayesgan