class Optimizer:
  def __init__(self, p, lr, beta):
    self.p, self.lr, self.beta = p, lr, beta
    self.p, self.lr, self.beta = p, lr, beta
    self.v = [0] * len(self.p)

  def zero(self):
    for p in self.p: p.grad = 0


class SGD(Optimizer):
  def __init__(self, p, lr, beta=0.9):
    super().__init__(p, lr, beta)

  def step(self):
    for i, p in enumerate(self.p):
      self.v[i] = self.beta * self.v[i] + self.lr * p.grad
      p.val -= self.v[i]
