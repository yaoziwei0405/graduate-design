# 多重先验模型的最优解

import tushare as ts
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy.optimize as sco
import sympy

symbols = ['郑州煤电', '浦发银行', '青岛啤酒', '银亿股份']
data = pd.DataFrame()
data0 = ts.get_hist_data('600121', '2019-01-01', '2019-12-31')
data0 = data0['close']  # 郑州煤电收盘价数据
data0 = data0[::-1]  # 按日期从小到大排序
data['郑州煤电'] = data0

data1 = ts.get_hist_data('600000', '2019-01-01', '2019-12-31')
data1 = data1['close']  # 浦发银行收盘价数据
data1 = data1[::-1]
data['浦发银行'] = data1

data2 = ts.get_hist_data('600600', '2019-01-01', '2019-12-31')
data2 = data2['close']  # 青岛啤酒收盘价数据
data2 = data2[::-1]
data['青岛啤酒'] = data2

data3 = ts.get_hist_data('000981', '2019-01-01', '2019-12-31')
data3 = data3['close']  # 银亿股份收盘价数据
data3 = data3[::-1]
data['银亿股份'] = data3

# 数据清理
data = data.dropna()
data.head()


# 其余数据
T = 243 # 每一个风险资产（股票）2019年内共有243天的股价数据
N = 4 #总共有4个无风险资产
risk_Averse = 2 #取风险厌恶系数为2

# 计算不同股票的均值、协方差和相关系数
returns = np.log(data / data.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns_mean = returns.mean()
returns_cov = returns.cov()
returns.corr()

# 用蒙特卡洛模拟产生大量随机组合
noa = 4
port_returns = []
port_variance = []
for p in range(2000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    port_returns.append(np.sum(returns.mean() * 243 * weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 243, weights))))
port_returns = np.array(port_returns)
port_variance = np.array(port_variance)

## 资产预期收益率具有不同置信区间的情况：
alpha1 = 0.01 #百分之99置信区间
alpha2 = 0.1 #百分之90置信区间
alpha3 = 0.2 #百分之80置信区间
alpha = np.array([alpha1,alpha2,alpha3])

Alpha = alpha * (T - 1) * N / (T * (T - N))

def statistic(weights,i):
    weights = np.array(weights)
    port_returns = np.sum(returns.mean() * weights) * 243
    port_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 243, weights)))
    max_eq = port_returns - (risk_Averse/2) * port_variance - np.sqrt(Alpha[i] * port_variance)
    return np.array([port_returns, port_variance, port_returns / port_variance,-max_eq])

def eq_max1(weights):
    return statistic(weights,0)[3]

def eq_max2(weights):
    return statistic(weights,1)[3]

def eq_max3(weights):
    return statistic(weights,2)[3]
# 约束是所有参数权重组合为1
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))

opts1 = sco.minimize(eq_max1, noa * [1. /noa], method='SLSQP', bounds=bnds, constraints=cons)
print(f"多重先验模型下的99%置信区间投资组合权重：{opts1['x'].round(5)}")
print(f"相应的预期收益率，波动率和夏普比：{statistic(opts1['x'].round(5),0)}")

opts2 = sco.minimize(eq_max2, noa * [1. /noa], method='SLSQP', bounds=bnds, constraints=cons)
print(f"多重先验模型下的90%置信区间投资组合权重：{opts2['x'].round(5)}")
print(f"相应的预期收益率，波动率和夏普比：{statistic(opts2['x'].round(5),1)}")

opts3 = sco.minimize(eq_max3, noa * [1. /noa], method='SLSQP', bounds=bnds, constraints=cons)
print(f"多重先验模型下的80%置信区间投资组合权重：{opts3['x'].round(5)}")
print(f"相应的预期收益率，波动率和夏普比：{statistic(opts3['x'].round(5),2)}")

print(returns.corr())

