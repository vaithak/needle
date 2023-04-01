"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=in_features,
                                                     fan_out=out_features,
                                                     device=device,
                                                     dtype=dtype,
                                                     requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(fan_in=out_features,
                                                       fan_out=1,
                                                       device=device,
                                                       dtype=dtype,
                                                       requires_grad=True))
            self.bias.data = ops.reshape(self.bias.data, (1, out_features))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        unbiased = ops.matmul(X, self.weight)
        if self.bias is None:
            return unbiased
        return unbiased + ops.broadcast_to(self.bias, unbiased.shape)
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        logsumexp = ops.logsumexp(logits, axes=1)
        zy = ops.summation(logits * init.one_hot(logits.shape[1],
                                                 y,
                                                 device=logits.device,
                                                 dtype=logits.dtype,
                                                 requires_grad=False), axes=1)
        total_loss = ops.summation(logsumexp - zy)
        return total_loss / np.float32(logits.shape[0]) 
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            col_mean = ops.summation(x, axes=0) / np.float32(x.shape[0])
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * col_mean.reshape(self.dim).data
            col_mean = col_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
            diff_from_mean = x - col_mean

            col_var = ops.summation(diff_from_mean ** 2, axes=0) / np.float32(x.shape[0])
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * col_var.reshape(self.dim).data
            col_var = col_var.reshape((1, x.shape[1])).broadcast_to(x.shape)
            return self.bias.broadcast_to(x.shape) + self.weight.broadcast_to(x.shape) * (diff_from_mean) / (col_var + self.eps)**0.5
        else:
            diff_from_mean = x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape).data
            std = (self.running_var.reshape((1, self.dim)).broadcast_to(x.shape).data + self.eps)**0.5
            return self.bias.broadcast_to(x.shape) + self.weight.broadcast_to(x.shape) * diff_from_mean / std
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        row_mean = ops.summation(x, axes=1) / np.float32(x.shape[1])
        row_mean = row_mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        diff_from_mean = x - row_mean

        row_var = ops.summation(diff_from_mean ** 2, axes=1) / np.float32(x.shape[1])
        row_var = row_var.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        return self.bias.broadcast_to(x.shape) + self.weight.broadcast_to(x.shape) * (diff_from_mean) / (row_var + self.eps)**0.5
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            return x * mask / (1-self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



