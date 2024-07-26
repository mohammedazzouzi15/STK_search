# InstanceNorm

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Segnn](./index.md#segnn) / InstanceNorm

> Auto-generated documentation for [src.stk_search.geom3d.models.SEGNN.instance_norm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/instance_norm.py) module.

- [InstanceNorm](#instancenorm)
  - [InstanceNorm](#instancenorm-1)
    - [InstanceNorm().forward](#instancenorm()forward)

## InstanceNorm

[Show source in instance_norm.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/instance_norm.py#L7)

Instance normalization for orthonormal representations
It normalizes by the norm of the representations.
Note that the norm is invariant only for orthonormal representations.
Irreducible representations `wigner_D` are orthonormal.
Parameters
----------
irreps : `Irreps`
    representation
eps : float
    avoid division by zero when we normalize by the variance
affine : bool
    do we have weight and bias parameters
reduce : {'mean', 'max'}
    method used to reduce

#### Signature

```python
class InstanceNorm(nn.Module):
    def __init__(
        self, irreps, eps=1e-05, affine=True, reduce="mean", normalization="component"
    ): ...
```

### InstanceNorm().forward

[Show source in instance_norm.py:51](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/instance_norm.py#L51)

evaluate
Parameters
----------
input : `torch.Tensor`
    tensor of shape ``(batch, ..., irreps.dim)``
Returns
-------
`torch.Tensor`
    tensor of shape ``(batch, ..., irreps.dim)``

#### Signature

```python
def forward(self, input, batch): ...
```