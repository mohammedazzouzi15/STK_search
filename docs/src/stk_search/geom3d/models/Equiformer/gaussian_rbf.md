# Gaussian Rbf

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Gaussian Rbf

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.gaussian_rbf](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/gaussian_rbf.py) module.

- [Gaussian Rbf](#gaussian-rbf)
  - [GaussianRadialBasisLayer](#gaussianradialbasislayer)
    - [GaussianRadialBasisLayer().extra_repr](#gaussianradialbasislayer()extra_repr)
    - [GaussianRadialBasisLayer().forward](#gaussianradialbasislayer()forward)
  - [gaussian](#gaussian)

## GaussianRadialBasisLayer

[Show source in gaussian_rbf.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/gaussian_rbf.py#L12)

#### Signature

```python
class GaussianRadialBasisLayer(torch.nn.Module):
    def __init__(self, num_basis, cutoff): ...
```

### GaussianRadialBasisLayer().extra_repr

[Show source in gaussian_rbf.py:43](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/gaussian_rbf.py#L43)

#### Signature

```python
def extra_repr(self): ...
```

### GaussianRadialBasisLayer().forward

[Show source in gaussian_rbf.py:32](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/gaussian_rbf.py#L32)

#### Signature

```python
def forward(self, dist, node_atom=None, edge_src=None, edge_dst=None): ...
```



## gaussian

[Show source in gaussian_rbf.py:4](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/gaussian_rbf.py#L4)

#### Signature

```python
@torch.jit.script
def gaussian(x, mean, std): ...
```