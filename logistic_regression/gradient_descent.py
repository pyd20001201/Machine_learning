import numpy as np
import matplotlib.pyplot as plt


def min_max_normalize(_x):
    """
    Min-mix normalize to get normalized x.
    When testing new data, it also need to be normalized.
    """
    _x_min, _x_max = np.min(_x, axis=0), np.max(_x, axis=0)
    x = (_x - _x_min) / (_x_max - _x_min)
    return x


def sigmoid(s):
    return 1/(1+np.exp(-s))


def _plot(x, y):
    plt.title('Gradient descent')
    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.ion()
    x_0, x_1 = x[np.where(y == 0)[0]], x[np.where(y == 1)[0]]
    plt.scatter(x_0[:, 1], x_0[:, 2], marker='o', facecolors='none', edgecolors='r')
    plt.scatter(x_1[:, 1], x_1[:, 2], marker='*', facecolors='none', edgecolors='r')
    line, = plt.plot([], [], 'g-')
    return line


def estimation(x, y , pred):
    loss = -np.sum(y * np.log2(pred) + (1-y) * np.log2(1-pred))
    grad = np.matmul(x.T, pred-y)
    return loss, grad


def draw(x, y, theta, line):
    min_x1, max_x1 = np.min(x, axis=0)[1] - 0.1, np.max(x, axis=0)[1] + 0.1
    # construct two points
    line_x = np.array([min_x1, max_x1])
    theta1 = theta.flatten()
    # calculate the x[:, 2] of these two points
    line_y = - (theta1[0] + theta1[1] * line_x) / theta1[2]
    line.set_data(line_x, line_y)


def gradient_descent(x, y, theta, learning_rate=0.05, iter_times=400):
     line = _plot(x, y)
     for i in range(iter_times):
         pred = sigmoid(np.matmul(x, theta))
         loss, grad = estimation(x, y, pred)
         theta -= learning_rate*grad
         print("- iter {}, loss {:.8f}".format(i + 1, loss))
         draw(x, y, theta, line)
         plt.pause(0.01)
     plt.ioff()
     plt.show()


_x = np.loadtxt(r"d:/exam_x.dat", dtype=float, ndmin=2)
y = np.loadtxt(r"d:/exam_y.dat", dtype=float, ndmin=2).astype(int)
_x = np.array(_x)
y = np.array(y)

x = min_max_normalize(_x)
x = np.hstack((np.ones((x.shape[0], 1)), x))
theta = np.zeros((x.shape[1], 1))
gradient_descent(x, y, theta)




