from autonn import *
import matplotlib.pyplot as plt


nn = NN((3, 4, 4), (sigmoid, relu), 1e-2)

losses = []

print(f"before\n{nn([4, 2, 1, 8])}\n")

for _ in range(50):
  loss = mse(nn([4, 2, 1, 8]), [0, 1, 1, 0])
  losses.append(loss.val)
  loss.backward()
  nn.step()

print(f"after\n{nn([4, 2, 1, 8])}")

fig = plt.figure(figsize=(8, 8))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(losses)
plt.savefig("loss.png")
