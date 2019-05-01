import numpy as np
from matplotlib import pyplot as plt


def mean_squared_error(points, m, b):
    error = 0
    N = len(points)
    for i in range(N):
        x = points[i, 0]
        y = points[i, 1]
        error += (y - (m * x + b)) ** 2

    return error / N


def gradient_step(points, m, b, learning_rate):
    b_gradient = 0
    m_gradient = 0

    N = len(points)

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - (m * x + b))
        m_gradient += -(2/N) * x * (y - (m * x + b))

    b = b - b_gradient * learning_rate
    m = m - m_gradient * learning_rate

    return m, b


def gradient_descend(points, m, b, iterations, learning_rate):
    for i in range(iterations):
        m, b = gradient_step(points, m, b, learning_rate)

    return m, b


def run():
    points = np.genfromtxt('data.csv', delimiter=',')
    m = 0
    b = 0
    iterations = 1000
    learning_rate = 0.0001

    m, b = gradient_descend(points, m, b, iterations, learning_rate)

    plt.plot(points, 'bo')
    plt.plot([x * m + b for x in range(100)])
    plt.show()


run()
