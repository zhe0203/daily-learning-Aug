# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.interpolate import lagrange  # 拉格朗日函数

# 设置样本
df = pd.DataFrame({'id':pd.date_range('20100101',periods=20),'value':np.array(np.random.randn(20) * 20)})
print(df.iloc[10,1])
df.iloc[10,1] = np.nan
print(df['value'][2])


def ploy(s,n,k=6):
    y = s[list(range(n-k,n))+list(range(n+1,n+1+k))] # 取数
    y = y[y.notnull()]
    return lagrange(y.index,list(y))(n)
for i in df.columns:
    for j in range(len(df)):
        if(df[i].isnull())[j]:
            df[i][j] = ploy(df[i],j)
print(df)



# 使用scipy库进行插值
from scipy.interpolate import interp1d  # 一维数组的插值
import matplotlib.pyplot as plt

# 构造数据
measured_time = np.linspace(0,1,10)
noise = (np.random.random(10)*2-1) * 1e-1
# print(noise)
# 构建正弦曲线
measures = np.sin(2 * np.pi * measured_time) + noise
# print(measures)
# 构建一个线性插值函数
linear_interp = interp1d(measured_time,measures)
# 线性插值的结果
computed_time = np.linspace(0,1,1000)
linear_results = linear_interp(computed_time)  # 计算想要的插值结果

# 采用3次拟合的插值方法
cubic_interp = interp1d(measured_time, measures, kind='cubic')
cubic_results = cubic_interp(computed_time)

# 设置kind的方法为nearest
nearest_interp = interp1d(measured_time,measures,kind='nearest')
nearest_result = nearest_interp(computed_time)

# 进行绘图对比分析
fig = plt.figure()
ax = fig.add_subplot()
plt.plot(measured_time,measures,'o-',label='no',color='red')
plt.plot(computed_time,linear_results,label='linear',color='#FF6347')
plt.plot(computed_time,cubic_results,'--',label='cubic',color='#00FA9A')
plt.plot(computed_time,nearest_result,label='nearest',color='#FFC0CB')
plt.legend(loc='upper right')
plt.show()


########################### 使用RBF进行插值 ###########################
# 1-d example
import numpy as np
from scipy.interpolate import Rbf,InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

x = np.linspace(0,10,9)
print(x)
y = np.sin(x)
xi = np.linspace(0,10,1001)
# use fitpack2 method
ius = InterpolatedUnivariateSpline(x,y)
yi = ius(xi)
plt.subplot(2,1,1)
plt.plot(x,y,'bo')
plt.plot(xi,yi,'g')
plt.plot(xi,np.sin(xi),'r')

# use RBF
rbf = Rbf(x,y)
fi = rbf(xi)
plt.subplot(2,1,2)
plt.plot(x,y,'bo')
plt.plot(xi,yi,'g')
plt.plot(xi,np.sin(xi),'r')
plt.show()

# 2-D example
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from matplotlib import cm
x = np.random.rand(100)*4-2
y = np.random.rand(100)*4-2
z = x*np.exp(-x**2-y**2)
ti = np.linspace(-2,2,100)
xi,yi = np.meshgrid(ti,ti)

rbf = Rbf(x,y,z,epsilon=2)
zi = rbf(xi,yi)

plt.subplot()
plt.pcolor(xi,yi,zi,cmap=cm.jet)
plt.scatter(x,y,100,z,cmap=cm.jet)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.colorbar()
plt.show()
