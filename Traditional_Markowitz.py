import tushare as ts
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as scs
import matplotlib.pyplot as plt
import scipy.optimize as sco

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False


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


# 规范后的时序数据
#(data / data.iloc[0] * 100).plot(figsize=(8, 4))
#plt.show()

# 计算不同股票的均值、协方差和相关系数
returns = np.log(data / data.shift(1))  # shift(1)：水平向下移动一个单位（默认axis=0）
returns.mean() * 252
returns.cov()
returns.corr()
# 由此可见，各股票之间的相关系数不太大（大嘘），可以做投资组合


# 随机分配初始权重
# 假设不允许卖空
noa = 4
weights = np.random.random(noa)
weights /= np.sum(weights)

# 计算预期组合收益、组合方差和组合标准差
np.sum(returns.mean() * weights)  # 预期收益
np.dot(weights.T, np.dot(returns.cov(), weights))  # 方差
np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))  # 标准差

# 用蒙特卡洛模拟产生大量随机组合

port_returns = []
port_variance = []
for p in range(10000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    port_returns.append(np.sum(returns.mean() * 243 * weights))
    port_variance.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 243, weights))))
port_returns = np.array(port_returns)
port_variance = np.array(port_variance)
# 无风险利率设定为1.5%
risk_free = 0.015
plt.figure(figsize=(8, 4))
plt.scatter(port_variance, port_returns, c=(port_returns - risk_free) / port_variance, marker='o')
plt.grid(True)
plt.xlabel('期望波动率')
plt.ylabel('期望收益')
plt.colorbar(label='夏普比率')
plt.show()

# 夏普比最大的投资组合优化
def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(returns.mean() * weights) * 243
    port_variance = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 243, weights)))
    return np.array([port_returns, port_variance, port_returns / port_variance])


def min_sharpe(weights):
    return -statistics(weights)[2]
# 约束是所有参数权重组合为1
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
opts = sco.minimize(min_sharpe, noa * [1. /noa], method='SLSQP', bounds=bnds, constraints=cons)
opts
print(f"夏普比率最大的投资组合权重：{opts['x'].round(3)}")  # 取三位小数
print(f"相应的预期收益率，波动率和夏普比：{statistics(opts['x']).round(3)}")  # 预期收益率，波动率和夏普比


# 方差最小的投资组合优化
def min_variance(weights):
    return statistics(weights)[1]


optv = sco.minimize(min_variance, noa * [1. / noa], method='SLSQP', bounds=bnds, constraints=cons)
print(f"方差最小的投资组合权重：{optv['x'].round(3)}")
print(f"相应的预期收益率，波动率和夏普比：{statistics(optv['x'].round(3))}")

print(f"各股票日收益率：{returns.mean()}")
print(f"各股票年化收益率：{returns.mean()*243}")
print(f"各股票协方差：{returns.cov()}")
print(f"各股票相关系数：{returns.corr()}")



