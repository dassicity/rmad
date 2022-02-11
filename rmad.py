from math import log


class Var:
  def __init__(self, val, dag=False, fn=lambda *_: None):
    self.val, self.dag, self.fn = val, dag, fn
    self.grad = 0

  def __repr__(self):
    return "Var({:.4f})".format(self.val)

  def __mul__(self, o):
    o = o if isinstance(o, Var) else Var(o)
    def fn(grad):
      if self.dag: self.back(grad * o.val)
      if o.dag: o.back(grad * self.val)
    return Var(self.val * o.val, self.dag | o.dag, fn)

  def __neg__(self):
    return self * -1

  def __add__(self, o):
    o = o if isinstance(o, Var) else Var(o)
    def fn(grad):
      if self.dag: self.back(grad)
      if o.dag: o.back(grad)
    return Var(self.val + o.val, self.dag | o.dag, fn)

  def __sub__(self, o):
    o = o if isinstance(o, Var) else Var(o)
    def fn(grad):
      if self.dag: self.back(grad)
      if o.dag: o.back(-grad)
    return Var(self.val - o.val, self.dag | o.dag, fn)

  def __truediv__(self, o):
    o = o if isinstance(o, Var) else Var(o)
    def fn(grad):
      if self.dag: self.back(grad * 1 / o.val)
      if o.dag: o.back(grad * -self.val / (o.val * o.val))
    return Var(self.val / o.val, self.dag | o.dag, fn)

  def __pow__(self, o):
    o = o if isinstance(o, Var) else Var(o)
    def fn(grad):
      if self.dag: self.back(grad * o.val * self.val ** (o.val - 1))
      if o.dag: o.back(grad * log(self.val) * self.val ** o.val)
    return Var(self.val ** o.val, self.dag | o.dag, fn)

  def log(self):
    def fn(grad):
      if self.dag: self.back(grad * 1 / self.val)
    return Var(log(self.val), self.dag, fn)

  def __radd__(self, o):
    return self + o

  def __rsub__(self, o):
    return -self + o

  def __rmul__(self, o):
    return self * o

  def __rtruediv__(self, o):
    return o * self ** -1

  def __rpow__(self, o):
    return o ** self if isinstance(o, Var) else Var(o) ** self

  def __gt__(self, o):
    return self.val > o

  def __lt__(self, o):
    return self.val < o

  def back(self, grad=1):
    self.grad += grad
    self.fn(self.grad)
