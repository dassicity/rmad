from rmad import Var 
from random import random


def rand(n):
  return [Var(random() / n, True) for _ in range(n)]


class NN:
  def __init__(self, ds, fs):
    self.ds, self.fs = ds, fs
    self.w = [[rand(m) for _ in range(n)] for m, n in zip(ds[:-1], ds[1:])]
    self.b = [rand(n) for n in ds[1:]]
    self.p = sum((sum(w, []) for w in self.w), []) + sum(self.b, [])

  def __call__(self, x):
    for i, (w, b, f) in enumerate(zip(self.w, self.b, self.fs)):
      wx = [sum(a * b for a, b in zip(x, r)) for r in w]
      x = [f(a) for a in [c + d for c, d in zip(wx, b)]]
    return x
