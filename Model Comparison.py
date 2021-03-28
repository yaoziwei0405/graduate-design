# 两个策略在2020内的表现

import tushare as ts
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy.optimize as sco

symbols = ['郑州煤电', '浦发银行', '青岛啤酒', '银亿股份']
data_1month = pd.DataFrame()
data_3month = pd.DataFrame()
data_6month = pd.DataFrame()
data_9month = pd.DataFrame()
data_12month = pd.DataFrame()

## 一个月的数据
data0 = ts.get_hist_data('600121', '2020-01-01', '2020-01-31')
data0 = data0['close']  # 郑州煤电收盘价数据
data0 = data0[::-1]  # 按日期从小到大排序
data_1month['600121'] = data0

data1 = ts.get_hist_data('600000', '2020-01-01', '2020-01-31')
data1 = data1['close']  # 浦发银行收盘价数据
data1 = data1[::-1]
data_1month['600000'] = data1

data2 = ts.get_hist_data('600600', '2020-01-01', '2020-01-31')
data2 = data2['close']  # 青岛啤酒收盘价数据
data2 = data2[::-1]
data_1month['000980'] = data2

data3 = ts.get_hist_data('000981', '2020-01-01', '2020-01-31')
data3 = data3['close']  # 银亿股份收盘价数据
data3 = data3[::-1]
data_1month['000981'] = data3


## 三个月的数据
data0 = ts.get_hist_data('600121', '2020-01-01', '2020-03-31')
data0 = data0['close']  # 郑州煤电收盘价数据
data0 = data0[::-1]  # 按日期从小到大排序
data_3month['600121'] = data0

data1 = ts.get_hist_data('600000', '2020-01-01', '2020-03-31')
data1 = data1['close']  # 浦发银行收盘价数据
data1 = data1[::-1]
data_3month['600000'] = data1

data2 = ts.get_hist_data('600600', '2020-01-01', '2020-03-31')
data2 = data2['close']  # 青岛啤酒收盘价数据
data2 = data2[::-1]
data_3month['000980'] = data2

data3 = ts.get_hist_data('000981', '2020-01-01', '2020-03-31')
data3 = data3['close']  # 银亿股份收盘价数据
data3 = data3[::-1]
data_3month['000981'] = data3


## 六个月的数据
data0 = ts.get_hist_data('600121', '2020-01-01', '2020-06-30')
data0 = data0['close']  # 郑州煤电收盘价数据
data0 = data0[::-1]  # 按日期从小到大排序
data_6month['600121'] = data0

data1 = ts.get_hist_data('600000', '2020-01-01', '2020-06-30')
data1 = data1['close']  # 浦发银行收盘价数据
data1 = data1[::-1]
data_6month['600000'] = data1

data2 = ts.get_hist_data('600600', '2020-01-01', '2020-06-30')
data2 = data2['close']  # 青岛啤酒收盘价数据
data2 = data2[::-1]
data_6month['000980'] = data2

data3 = ts.get_hist_data('000981', '2020-01-01', '2020-06-30')
data3 = data3['close']  # 银亿股份收盘价数据
data3 = data3[::-1]
data_6month['000981'] = data3

# 九个月的数据
data0 = ts.get_hist_data('600121', '2020-01-01', '2020-09-30')
data0 = data0['close']  # 郑州煤电收盘价数据
data0 = data0[::-1]  # 按日期从小到大排序
data_9month['600121'] = data0

data1 = ts.get_hist_data('600000', '2020-01-01', '2020-09-30')
data1 = data1['close']  # 浦发银行收盘价数据
data1 = data1[::-1]
data_9month['600000'] = data1

data2 = ts.get_hist_data('600600', '2020-01-01', '2020-09-30')
data2 = data2['close']  # 青岛啤酒收盘价数据
data2 = data2[::-1]
data_9month['000980'] = data2

data3 = ts.get_hist_data('000981', '2020-01-01', '2020-09-30')
data3 = data3['close']  # 银亿股份收盘价数据
data3 = data3[::-1]
data_9month['000981'] = data3

# 十二个月的数据
data0 = ts.get_hist_data('600121', '2020-01-01', '2020-12-31')
data0 = data0['close']  # 郑州煤电收盘价数据
data0 = data0[::-1]  # 按日期从小到大排序
data_12month['600121'] = data0

data1 = ts.get_hist_data('600000', '2020-01-01', '2020-12-31')
data1 = data1['close']  # 浦发银行收盘价数据
data1 = data1[::-1]
data_12month['600000'] = data1

data2 = ts.get_hist_data('600600', '2020-01-01', '2020-12-31')
data2 = data2['close']  # 青岛啤酒收盘价数据
data2 = data2[::-1]
data_12month['000980'] = data2

data3 = ts.get_hist_data('000981', '2020-01-01', '2020-12-31')
data3 = data3['close']  # 银亿股份收盘价数据
data3 = data3[::-1]
data_12month['000981'] = data3

# 数据清理
data_3month = data_3month.dropna()
data_6month = data_6month.dropna()
data_9month = data_9month.dropna()
data_12month = data_12month.dropna()

# 规范后的2020年的时序数据
(data_12month / data_12month.iloc[0] * 100).plot(figsize=(8, 4))
plt.show()

# 计算不同股票的均值、协方差和相关系数
returns_1month = np.log(data_1month / data_1month.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns_1month_mean = returns_1month.mean() * returns_1month.shape[0]

returns_3month = np.log(data_3month / data_3month.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns_3month_mean = returns_3month.mean() * returns_3month.shape[0]

returns_6month = np.log(data_6month / data_6month.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns_6month_mean = returns_6month.mean() * returns_6month.shape[0]

returns_9month = np.log(data_9month / data_9month.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns_9month_mean = returns_9month.mean() * returns_9month.shape[0]

returns_12month = np.log(data_12month / data_12month.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns_12month_mean = returns_12month.mean() * returns_12month.shape[0]

# 由此可见，各股票之间的相关系数不太大（大嘘），可以做投资组合

# 由Traditional——Markowitz.py与Multi-prior approach.p得出的权重数据
weight_T_maxsharpe = np.array([0, 0.679, 0.321, 0])
weight_T_minvariance = np.array([0.044,0.738,0.147,0.071])
#weight_M_99 = np.array([0,0.621,0.379,0])
#weight_M_90 = np.array([0,0.663,0.337,0])
#weight_M_80 = np.array([0,0.681,0.319,0])

weight_M_99 = np.array([0,0.598,0.402,0])
weight_M_90 = np.array([0,0.606,0.394,0])
weight_M_80 = np.array([0,0.611,0.389,0])


# 计算预期组合收益、组合方差和组合标准差
print("小alpha ！= 大alpha的情况")
## 一个月的收益率
a0 = np.sum(returns_1month_mean * weight_T_maxsharpe)  # 传统马科维茨模型,最大夏普比率模型
b0 = np.sum(returns_1month_mean * weight_T_minvariance) # 传统马科维茨模型，最小方差模型
c0 = np.sum(returns_1month_mean * weight_M_99) #多重先验模型联合分布，99%置信区间
d0 = np.sum(returns_1month_mean * weight_M_90) #多重先验模型联合分布，90%置信区间
e0 = np.sum(returns_1month_mean * weight_M_80) #多重先验模型联合分布，80%执行区间
print(f"一个月的收益率：{a0} , {b0} , {c0} , {d0} , {e0}")

## 三个月的收益率
a1 = np.sum(returns_3month_mean * weight_T_maxsharpe)  # 传统马科维茨模型,最大夏普比率模型
b1 = np.sum(returns_3month_mean * weight_T_minvariance) # 传统马科维茨模型，最小方差模型
c1 = np.sum(returns_3month_mean * weight_M_99) #多重先验模型联合分布，99%置信区间
d1 = np.sum(returns_3month_mean * weight_M_90) #多重先验模型联合分布，90%置信区间
e1 = np.sum(returns_3month_mean * weight_M_80) #多重先验模型联合分布，80%执行区间
print(f"三个月的收益率：{a1} , {b1} , {c1} , {d1} , {e1}")

## 六个月的收益率
a2 = np.sum(returns_6month_mean * weight_T_maxsharpe)  # 传统马科维茨模型,最大夏普比率模型
b2 = np.sum(returns_6month_mean * weight_T_minvariance) # 传统马科维茨模型，最小方差模型
c2 = np.sum(returns_6month_mean * weight_M_99) #多重先验模型联合分布，99%置信区间
d2 = np.sum(returns_6month_mean * weight_M_90) #多重先验模型联合分布，90%置信区间
e2 = np.sum(returns_6month_mean * weight_M_80) #多重先验模型联合分布，80%执行区间
print(f"六个月的收益率：{a2} , {b2} , {c2} , {d2} , {e2}")

## 九个月的收益率
a3 = np.sum(returns_9month_mean * weight_T_maxsharpe)  # 传统马科维茨模型,最大夏普比率模型
b3 = np.sum(returns_9month_mean * weight_T_minvariance) # 传统马科维茨模型，最小方差模型
c3 = np.sum(returns_9month_mean * weight_M_99) #多重先验模型联合分布，99%置信区间
d3 = np.sum(returns_9month_mean * weight_M_90) #多重先验模型联合分布，90%置信区间
e3 = np.sum(returns_9month_mean * weight_M_80) #多重先验模型联合分布，80%执行区间
print(f"九个月的收益率：{a3} , {b3} , {c3} , {d3} , {e3}")

## 十二个月的收益率
a4 = np.sum(returns_12month_mean * weight_T_maxsharpe)  # 传统马科维茨模型,最大夏普比率模型
b4 = np.sum(returns_12month_mean * weight_T_minvariance) # 传统马科维茨模型，最小方差模型
c4 = np.sum(returns_12month_mean * weight_M_99) #多重先验模型联合分布，99%置信区间
d4 = np.sum(returns_12month_mean * weight_M_90) #多重先验模型联合分布，90%置信区间
e4 = np.sum(returns_12month_mean * weight_M_80) #多重先验模型联合分布，80%执行区间
print(f"十二个月的收益率：{a4} , {b4} , {c4} , {d4} , {e4}")

#输出结果







