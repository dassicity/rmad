from autodiff import *
from random import random
from math import e


def relu(x):
  return x * (x > 0)


def sigmoid(x):
  return 1 / (1 + e ** (-x))


def mul(w, x):
  return [sum(a * b for a, b in zip(x, r)) for r in w]


def randl(n):
  return [Var(random(), reqdf=True) for _ in range(n)]


def mse(p, y):
  return sum((b - a) ** 2 for a, b in zip(p, y))


class NN:
  def __init__(self, dims, fs, lr):
    self.dims = dims
    self.w = [[randl(m) for _ in range(n)] for m, n in zip(dims[:-1], dims[1:])]
    self.b = [randl(n) for n in dims[1:]]
    self.l = [[0] * n for n in dims]
    self.fs = fs
    self.lr = lr

  def zerograd(self):
    for i, (m, n) in enumerate(zip(self.dims[1:], self.dims[:-1])):
      for r in range(m):
        self.b[i][r].grad = 0
        for c in range(n):
          self.w[i][r][c].grad = 0

  def step(self):
    for i, (m, n) in enumerate(zip(self.dims[1:], self.dims[:-1])):
      for r in range(m):
        self.b[i][r].val -= self.lr * self.b[i][r].grad.val
        for c in range(n):
          self.w[i][r][c].val -= self.lr * self.w[i][r][c].grad.val
    self.zerograd()

  def __call__(self, x):
    self.l[0] = x
    for i, (w, b, f) in enumerate(zip(self.w, self.b, self.fs)):
      self.l[i + 1] = [f(a) for a in [c + d for c, d in zip(mul(w, self.l[i]), b)]]
    return self.l[-1]
