import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]
x = np.reshape(x, newshape=(14, 1))

y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.583, 7.971, 8.561, 10.000, 11.280, 12.900]
y = np.reshape(y, newshape=(14, 1))

# 调用模型
lr = LinearRegression()
# 训练模型
lr.fit(x, y)
# 计算R平方
xx = [2014]
xx = np.reshape(xx, newshape=(1, 1))
yy = lr.predict(xx)
print(yy)
print(lr.score(x, y))
# 计算y_hat
y_hat = lr.predict(x)
# 打印出图
plt.scatter(x, y)
plt.plot(x, y_hat)
plt.title('Housing price in Nanjing')
plt.xlabel('year')
plt.ylabel('price')
plt.show()
