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
