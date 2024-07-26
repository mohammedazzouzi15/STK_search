# Fast Layer Norm

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Fast Layer Norm

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.fast_layer_norm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_layer_norm.py) module.

- [Fast Layer Norm](#fast-layer-norm)
  - [EquivariantLayerNormFast](#equivariantlayernormfast)
    - [EquivariantLayerNormFast().forward](#equivariantlayernormfast()forward)

## EquivariantLayerNormFast

[Show source in fast_layer_norm.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_layer_norm.py#L9)

#### Signature

```python
class EquivariantLayerNormFast(nn.Module):
    def __init__(self, irreps, eps=1e-05, affine=True, normalization="component"): ...
```

### EquivariantLayerNormFast().forward

[Show source in fast_layer_norm.py:36](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_layer_norm.py#L36)

Use torch layer norm for scalar features.

#### Signature

```python
def forward(self, node_input, **kwargs): ...
```