from math import log


def ddiv(i, a, b):
    return -a.val / (b.val * b.val) if i == 1 else 1 / b.val


def dpow(i, a, b):
  if i == 1:
    return log(a.val) * a.val ** b.val  
  else:
    return b.val * a.val ** (b.val - 1)


class Var:
  def __init__(self, val, xs=(None, None), reqgrad=False, gradfn=lambda *_: 1):
    if isinstance(val, Var):
      raise TypeError("val must not be Var")
    self.val = val
    self.xs = xs
    self.grad = None
    self.reqgrad = reqgrad
    self.gradfn = gradfn

  def __repr__(self):
    return f"Var({self.val}" + (f", reqgrad={self.reqgrad})" if self.reqgrad else ')')

  def par(self, val, other, gradfn):
    return Var(val, (self, other), self.reqgrad | other.reqgrad, gradfn)

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
    return self.par(log(self.val), Var(None), gradfn=lambda _, x, __: 1 / x.val)

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

  def _backward_(self):
    for i, x in enumerate(self.xs):
      if x and x.reqgrad:
        dx = self.grad * self.gradfn(i, *self.xs)
        x.grad = dx if x.grad is None else x.grad + dx
        x._backward_()

  def backward(self):
    self.grad = Var(1)
    self._backward_()


def grad(f):
  def wrap(*xs):
    xs = tuple(Var(x.val, reqgrad=True) for x in xs)
    y = f(*xs)
    y.backward()
    return tuple(x.grad for x in xs)
  return wrap
