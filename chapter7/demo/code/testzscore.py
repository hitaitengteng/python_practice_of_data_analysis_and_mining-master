#-*- coding: utf-8 -*-
# 标准差标准化


"""
@project: qianshan
@author: aitengteng
@file: explore
@time: 2019-04-02 12:50:33
"""


# 引入数据包
import sys
import pandas as pd


# 提取参数
datafile = sys.argv[1]
#  F:/downloads/zscoredata.xls  # 需要传入的参数，待进行标准化的数据文件；

# 存储结果
zscoredfile = 'F:/downloads/tmp/zscoreddata.xls'  # 标准差化后的数据存储路径文件；

# 标准化处理
data = pd.read_excel(datafile)
# 实现了标准化变换，类似地可以实现任何想要的变换。
data = (data - data.mean(axis=0)) / (data.std(axis=0))
data.columns = ['Z' + i for i in data.columns]  # 表头重命名。

data.to_excel(zscoredfile, index=False)  # 数据写入
print('END')