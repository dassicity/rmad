# y = f(x1 ... xn)
# dL / dxi = (dL / dy) * (dy / dxi)
# where f is some primitive operation
# primitive ops: +, -, *, /, **, log, min, max
# above is all that's needed for scalar auto diff

# f(x1 ... xn) returns a Var
# whose children are x1 ... xn, and op is remembered


from math import log


f = {
    "add": lambda a, b: a + b, 
    "sub": lambda a, b: a - b, 
    "mul": lambda a, b: a * b, 
    "div": lambda a, b: a / b, 
    "exp": lambda a, b: a ** b
}

df = {
    "add": lambda *_: 1, 
    "sub": lambda i, *_: -1 if i == 1 else 1, 
    "mul": lambda i, a, b: a if i == 1 else b, 
    "div": lambda i, a, b: -a / (b * b) if i == 1 else 1 / b, 
    "exp": lambda i, a, b: b * a ** (b - 1) if i == 0 else log(a) * a ** b
}


class Var:
    def __init__(self, val, args=None, x=None, requires_grad=False, fn=None):
        self.val = val
        self.grad = 0
        self.requires_grad = requires_grad
        self.fn = fn
        self.args = args
        self.x = x

    def __repr__(self):
        fn = f", fn={self.fn})" if self.requires_grad else ")"
        return f"Var({self.val}" + fn

    def par(self, fn, *args):
        x = []
        requires_grad = False
        for arg in args:
            x.append(arg.val)
            requires_grad |= arg.requires_grad
        return Var(f[fn](*x), args, x, requires_grad, fn)

    def __add__(self, b):
        return self.par("add", self, b)

    def __sub__(self, b):
        return self.par("sub", self, b)

    def __mul__(self, b):
        return self.par("mul", self, b)

    def __truediv__(self, b):
        return self.par("div", self, b)

    def __neg__(self):
        return self.par("mul", self, -1)

    def __pow__(self, b):
        return self.par("exp", self, b)

    def __lt__(self, b):
        return self.val < b.val

    def __gt__(self, b):
        return self.val > b.val

    def __eq__(self, b):
        return self.val == b.val

    def _backward_(self):
        if self.args:
            for i, arg in enumerate(self.args):
                if arg.requires_grad:
                    arg.grad += self.grad * df[self.fn](i, *self.x)
                    arg._backward_()

    def backward(self):
        self.grad = 1
        self._backward_()
