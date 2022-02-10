# autodiff

![](kitten.jpg)

This is a smöl implementation of reverse-mode automatic differentiation.

`x = Var(i, param=True)` - this creates a scalar which'll be part of the compute graph for computing gradients. 

```
x = Var(3, param=True)
y = x ** 2 + math.e ** x
y.backward()
print(x.grad)
```

...returns derivative of `y` w.r.t `x`, which is ~26.0855. Derivative of `x ** 2 + e ** x` is `2 * x + e ** x` - substituting 3 in latter gives the same answer, so it works.

`autonn.py` implements a neural net using auto diff in `test.py`. Very simple to train, but very slow since it doesn't use NumPy

<p align="center"><img src="loss.png"></img></p>
