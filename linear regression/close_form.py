
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt(r'd:/price_x.txt', dtype=int, ndmin=2)
x = np.array(x)
y = np.loadtxt(r'd:/price_y.txt', dtype=float, ndmin=2)
y = np.array(y)

x = np.hstack((np.ones((x.shape[0], 1)), x))


def close_form(x, y):
    temp = np.linalg.inv(np.matmul(x.T, x))
    theta = np.matmul(np.matmul(temp, x.T), y)
    return theta


def prediction(x, theta):
    x = [1, x]
    x = np.array()
    y = np.matmul(x, theta)
    return y


theta = close_form(x, y)
y_hat = np.matmul(x, theta)
xx = np.loadtxt(r'd:/price_x.txt', dtype=int, ndmin=2)
plt.scatter(xx, y)
plt.plot(xx, y_hat)
plt.title('housing price prediction')
plt.xlabel('year')
plt.ylabel('price')
plt.show()
