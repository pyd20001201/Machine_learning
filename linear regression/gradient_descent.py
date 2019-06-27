import numpy as np
import matplotlib.pyplot as plt

x = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013]

y = [2.000, 2.500, 2.900, 3.147, 4.515, 4.903, 5.365, 5.704, 6.583, 7.971, 8.561, 10.000, 11.280, 12.900]


"""normalize the data"""
def zero_score_normalization(x):
    mean_x = np.mean(x)
    std_x = np.std(x)
    x = (x - mean_x)/std_x
    return x, mean_x, std_x


"""hypothesis function"""
def model(a, b, x):
    return a*x + b


def cost_function(a, b, x, y):
    n = 14
    return 0.5/n * (np.square(y - a*x - b)).sum()


def optimize(a, b, x, y):
    n = 14
    alpha = 1e-2                             
    y_hat = model(a, b, x)
    da = (1.0/n)*((y_hat-y)*x).sum()
    db = (1.0/n)*((y_hat-y).sum())
    a = a - alpha*da
    b = b - alpha*db
    return a, b


a = 0.0
b = 0.0


def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y)

    y_hat = model(a, b, x)
    cost = cost_function(a, b, x, y)
    plt.scatter(x, y)
    plt.plot(x, y_hat)
    return a, b


x, mean_x, std_x = zero_score_normalization(x)

a, b = iterate(a, b, x, y, 10000)

y1 = b + a * (2014-mean_x)/std_x

print("Predict result of year 2014: %.6f" % y1)
plt.show()
