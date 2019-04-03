#-*- coding: utf-8 -*-
# 标准差标准化

import pandas as pd

datafile = 'F:/downloads/zscoredata.xls'  # 需要进行标准化的数据文件；
zscoredfile = 'F:/downloads/tmp/zscoreddata.xls'  # 标准差化后的数据存储路径文件；

# 标准化处理
data = pd.read_excel(datafile)
# 简洁的语句实现了标准化变换，类似地可以实现任何想要的变换。
data = (data - data.mean(axis=0)) / (data.std(axis=0))
data.columns = ['Z' + i for i in data.columns]  # 表头重命名。

data.to_excel(zscoredfile, index=False)  # 数据写入
print('END')