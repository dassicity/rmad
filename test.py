from nn import NN
from opt import SGD
from random import random
import matplotlib.pyplot as plt


def mse(p, y):  # mean square error
  return sum((b - a) ** 2 for a, b in zip(p, y))


def relu(x):
  return x * (x > 0)


# define NN and optimizer
nn = NN((3, 5, 4), (relu, relu))
opt = SGD(nn.p, 5e-3, 0.9)

losses = []

# random dataset
xs = [[random() for _ in range(3)] for _ in range(4)]
ys = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]

# train
for _ in range(150):
  loss = sum(mse(nn(x), y) for x, y in zip(xs, ys))
  losses.append(loss.val)
  opt.zero()
  loss.back()
  opt.step()

# result
print("nn(x):", *[nn(x) for x in xs])
print("y:", *ys)

# plot
fig = plt.figure(figsize=(8, 8))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(losses)
plt.savefig("loss.png")
