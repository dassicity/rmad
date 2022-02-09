from math import log


def ddiv(i, a, b):
    return -a.val / (b.val * b.val) if i == 1 else 1 / b.val


def dpow(i, a, b):
  if i == 1:
    return log(a.val) * a.val ** b.val  
  else:
    return b.val * a.val ** (b.val - 1)


class Var:
  def __init__(self, val, xs=(None, None), reqdf=False, df=lambda *_: 1):
    self.val = val
    self.xs = xs
    self.grad = 0
    self.reqdf = reqdf
    self.df = df

  def __repr__(self):
    return f"Var({self.val}" + (f", reqdf={self.reqdf})" if self.reqdf else ')')

  def par(self, val, other, df):
    return Var(val, (self, other), self.reqdf | other.reqdf, df)

  def __mul__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.par(self.val * other.val, other, lambda i, a, b: a.val if i == 1 else b.val)

  def __neg__(self):
    return self * -1

  def __add__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.par(self.val + other.val, other, lambda *_: 1)

  def __sub__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.par(self.val - other.val, other, lambda i, *_: -1 if i == 1 else 1)

  def __truediv__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.par(self.val / other.val, other, ddiv)

  def __pow__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.par(self.val ** other.val, other, dpow)

  def log(self):
    return self.par(log(self.val), Var(None), df=lambda _, x, __: 1 / x.val)

  def __radd__(self, other):
    return self + other

  def __rsub__(self, other):
    return -self + other

  def __rmul__(self, other):
    return self * other

  def __rtruediv__(self, other):
    return other * self ** -1

  def __rpow__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return other ** self

  def __lt__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.val < other.val

  def __gt__(self, other):
    other = other if isinstance(other, Var) else Var(other)
    return self.val > other.val

  def _backward_(self):
    for i, x in enumerate(self.xs):
      if x and x.reqdf:
        x.grad = self.grad * self.df(i, *self.xs)
        x._backward_()

  def backward(self):
    self.grad = Var(1)
    self._backward_()
