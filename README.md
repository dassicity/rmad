# autodiff


`x = Var(i, requires_grad=True)` - this creates a scalar which'll be part of the compute graph for computing gradients. 

Very basic operations only - `+, -, *, ^, max, min`. Example below...

```
x = Var(3, requires_grad=True)
y = (x ** 2) + (Var(math.e) ** x)
y.backward()
print(x.grad)
```

...returns derivative of `y` w.r.t `x`. This is a very, very basic implementation of `torch.autograd`.
