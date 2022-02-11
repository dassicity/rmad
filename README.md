# rmad

<p align="center"><img height=500px width=500px src="kitten.jpg"></img></p>

This is a smöl implementation of reverse-mode automatic differentiation.

`x = Var(i, dag=True)` creates a `Var` instance, where `dag=True` means calling `Var.back` will compute the derivative of the objective (or loss) w.r.t `x`...

```
x = Var(3, dag=True)
y = x ** 2 + e ** x
y.back()
print(x.grad)
```

...returns the derivative of `y` w.r.t `x`. The `grad` decorator in `rmad.py` returns the derivative (or tuple of derivatives) directly.

```
def f(x):
  return x ** 2 + e ** x
  
df = grad(f)

print(*df(3))  # prints df/dx evaluated @ x = 3
```

`nn.py` implements a neural net using `rmad.py` in `test.py` using optimizer from `opt.py`.

<p align="center"><img src="loss.png"></img></p>
