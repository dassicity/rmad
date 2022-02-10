from autodiff import *
from random import random
from math import e


def relu(x):
  return x * (x > 0)


def sigmoid(x):
  return 1 / (1 + e ** (-x))


def mul(w, x):  # matmul
  return [sum(a * b for a, b in zip(x, r)) for r in w]


def randl(n):
  return [Var(random() / n, param=True) for _ in range(n)]


def mse(p, y):  # mean square error
  return sum((b - a) ** 2 for a, b in zip(p, y))


class NN:
  def __init__(self, dims, fs, lr):
    self.dims = dims  # layer sizes
    self.w = [[randl(m) for _ in range(n)] for m, n in zip(dims[:-1], dims[1:])]
    self.b = [randl(n) for n in dims[1:]]
    self.l = [[0] * n for n in dims]
    self.fs = fs  # activation functions
    self.lr = lr  # learning rate

  def infer(self, param):  # disable grad during inference
    param = not param
    for i, (m, n) in enumerate(zip(self.dims[1:], self.dims[:-1])):
      for r in range(m):
        self.b[i][r].param = param
        for c in range(n):
          self.w[i][r][c].param = param

  def zero(self):  # after update, zero gradients for next backprop
    for i, (m, n) in enumerate(zip(self.dims[1:], self.dims[:-1])):
      for r in range(m):
        self.b[i][r].grad = 0
        for c in range(n):
          self.w[i][r][c].grad = 0

  def step(self):  # param update function
    for i, (m, n) in enumerate(zip(self.dims[1:], self.dims[:-1])):
      for r in range(m):
        self.b[i][r].val -= self.lr * self.b[i][r].grad
        for c in range(n):
          self.w[i][r][c].val -= self.lr * self.w[i][r][c].grad
    self.zero()

  def __call__(self, x):  # forward pass
    self.l[0] = x
    for i, (w, b, f) in enumerate(zip(self.w, self.b, self.fs)):
      self.l[i + 1] = [f(a) for a in [c + d for c, d in zip(mul(w, self.l[i]), b)]]
    return self.l[-1]
