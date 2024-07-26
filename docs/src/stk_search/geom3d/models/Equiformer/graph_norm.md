# Graph Norm

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Graph Norm

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.graph_norm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/graph_norm.py) module.

- [Graph Norm](#graph-norm)
  - [EquivariantGraphNorm](#equivariantgraphnorm)
    - [EquivariantGraphNorm().forward](#equivariantgraphnorm()forward)
  - [EquivariantGraphNormV2](#equivariantgraphnormv2)
    - [EquivariantGraphNormV2().forward](#equivariantgraphnormv2()forward)

## EquivariantGraphNorm

[Show source in graph_norm.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/graph_norm.py#L9)

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
class EquivariantGraphNorm(nn.Module):
    def __init__(
        self, irreps, eps=1e-05, affine=True, reduce="mean", normalization="component"
    ): ...
```

### EquivariantGraphNorm().forward

[Show source in graph_norm.py:57](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/graph_norm.py#L57)

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



## EquivariantGraphNormV2

[Show source in graph_norm.py:137](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/graph_norm.py#L137)

#### Signature

```python
class EquivariantGraphNormV2(nn.Module):
    def __init__(
        self, irreps, eps=1e-05, affine=True, reduce="mean", normalization="component"
    ): ...
```

### EquivariantGraphNormV2().forward

[Show source in graph_norm.py:178](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/graph_norm.py#L178)

#### Signature

```python
def forward(self, node_input, batch, **kwargs): ...
```