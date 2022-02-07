from autodiff import *


w = Var(1, requires_grad=True)
b = Var(1, requires_grad=True)
x = [Var(i) for i in range(10)]
y = [Var(2 * i + 5) for i in range(10)]

def mse(p, y):
    return (y - p) ** Var(2)

for _ in range(1000):
    w.requires_grad, b.requires_grad = True, True
    p = [w * xi + b for xi in x]
    loss = sum([mse(yi, pi) for pi, yi in zip(p, y)], Var(0))
    loss.backward()
    w.requires_grad, b.requires_grad = False, False
    w -= Var(2e-3 * w.grad)
    b -= Var(2e-3 * b.grad)

print(f"w: {w}\tb: {b}")
