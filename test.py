from autonn import *
import matplotlib.pyplot as plt


nn = NN((3, 5, 4), (relu, relu), 2e-2)  # initialize neural net

losses = []

# create a random classification dataset
x = [[random() for _ in range(3)] for _ in range(4)]
y = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]

for _ in range(500):
  loss = sum(mse(nn(a), b) for a, b in zip(x, y))  # compute mse loss
  losses.append(loss.val)
  loss.backward()  # backprop
  nn.step()  # update parameters

# print nn prediction and target vals
print(f"nn(x): {[[z.val for z in nn(a)] for a in x]}\ny: {y}\n")

# plot losses
fig = plt.figure(figsize=(8, 8))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(losses)
plt.savefig("loss.png")
