# autodiff


`x = Var(i, requires_grad=True)` - this creates a scalar which'll be part of the compute graph for computing gradients. 

Very basic operations only - `+, -, *, ^, log, max, min`...
