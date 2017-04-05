# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def test():
    print"test"


def graph(x_range, w):
    x = np.array(x_range)
    y = my_formula(w, x)
    plt.plot(x, y)
    plt.show()


def my_formula(w, x):
    return (w[2]*x**2)+(x*w[1])-(w[0])

#graph(my_formula, range(-10, 11))


def f_d(x):
    h = 1e-3
    print (f(x+h)-f(x))/h


def f(a):
    return 2*(a**2)+3*a+5


def f_t(a):
    return 4*a+3

if __name__ == "__main__":
    print f_t(2)
    print f_d(2)
