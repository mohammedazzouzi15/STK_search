# Instance Norm

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Instance Norm

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.instance_norm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/instance_norm.py) module.

- [Instance Norm](#instance-norm)
  - [EquivariantInstanceNorm](#equivariantinstancenorm)
    - [EquivariantInstanceNorm().forward](#equivariantinstancenorm()forward)

## EquivariantInstanceNorm

[Show source in instance_norm.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/instance_norm.py#L9)

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
class EquivariantInstanceNorm(nn.Module):
    def __init__(
        self, irreps, eps=1e-05, affine=True, reduce="mean", normalization="component"
    ): ...
```

### EquivariantInstanceNorm().forward

[Show source in instance_norm.py:56](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/instance_norm.py#L56)

evaluate
Parameters
----------
node_input : `torch.Tensor`
    tensor of shape ``(batch, ..., irreps.dim)``
Returns
-------
`torch.Tensor`
    tensor of shape ``(batch, ..., irreps.dim)``

#### Signature

```python
def forward(self, node_input, batch, **kwargs): ...
```