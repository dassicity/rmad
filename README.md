# autodiff

![](kitten.jpg)

This is a smöl implementation of reverse-mode automatic differentiation.

`x = Var(i, requires_grad=True)` - this creates a scalar which'll be part of the compute graph for computing gradients. 

Very basic operations only allowed - `+, -, *, ^, max, min`. Example below...

```
x = Var(3, requires_grad=True)
y = x ** 2 + math.e ** x
y.backward()
print(x.grad)
```

...returns derivative of `y` w.r.t `x`, which is ~26.0855. Derivative of `x ** 2 + e ** x` is `2 * x + e ** x`; substituting 3 in latter gives the same answer, so it works. This is a very basic implementation. More complex functions and array functionalities have not been added. 

See Karpathy's [micrograd](https://github.com/karpathy/micrograd) for something that resembles PyTorch's Autograd more closely.
