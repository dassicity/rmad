from autodiff import *
from random import random


def rand(n):
  return [Var(random() / n, True) for _ in range(n)]


class Infer:
  def __init__(self, *nns):
    self.nns = nns
  def __enter__(self):
    for nn in self.nns: nn.infer(True)
  def __exit__(self, *_):
    for nn in self.nns: nn.infer(False)


class NN:
  def __init__(self, ds, fs, lr):
    self.ds, self.fs, self.lr = ds, fs, lr
    self.w = [[rand(m) for _ in range(n)] for m, n in zip(ds[:-1], ds[1:])]
    self.b = [rand(n) for n in ds[1:]]

  def zero(self):
    for i, (m, n) in enumerate(zip(self.ds[1:], self.ds[:-1])):
      for r in range(m):
        self.b[i][r].grad = 0
        for c in range(n):
          self.w[i][r][c].grad = 0

  def infer(self, dag):
    dag = not dag
    for i, (m, n) in enumerate(zip(self.ds[1:], self.ds[:-1])):
      for r in range(m):
        self.b[i][r].dag = dag
        for c in range(n):
          self.w[i][r][c].dag = dag

  def step(self):
    for i, (m, n) in enumerate(zip(self.ds[1:], self.ds[:-1])):
      for r in range(m):
        self.b[i][r].val -= self.lr * self.b[i][r].grad
        for c in range(n):
          self.w[i][r][c].val -= self.lr * self.w[i][r][c].grad
    self.zero()

  def __call__(self, x):
    for i, (w, b, f) in enumerate(zip(self.w, self.b, self.fs)):
      wx = [sum(a * b for a, b in zip(x, r)) for r in w]
      x = [f(a) for a in [c + d for c, d in zip(wx, b)]]
    return x
