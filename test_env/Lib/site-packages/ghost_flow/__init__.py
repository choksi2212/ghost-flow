"""
GhostFlow - Blazingly Fast Machine Learning Framework

A production-ready ML framework built in Rust, providing Python bindings
for maximum performance while maintaining a Pythonic API.

Example:
    >>> import ghost_flow as gf
    >>> x = gf.Tensor.randn([32, 784])
    >>> y = gf.Tensor.randn([784, 10])
    >>> z = x @ y  # Matrix multiplication
"""

from ._ghost_flow import Tensor, nn, __version__

__all__ = ["Tensor", "nn", "__version__"]

# Convenience aliases
randn = Tensor.randn
rand = Tensor.rand
zeros = Tensor.zeros
ones = Tensor.ones
