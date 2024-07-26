# Tfn

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Tfn

> Auto-generated documentation for [src.stk_search.geom3d.models.TFN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN.py) module.

- [Tfn](#tfn)
  - [TFN](#tfn)
    - [TFN().forward](#tfn()forward)

## TFN

[Show source in TFN.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN.py#L13)

#### Signature

```python
class TFN(nn.Module):
    def __init__(
        self,
        num_layers,
        atom_feature_size,
        num_channels,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
    ): ...
```

### TFN().forward

[Show source in TFN.py:62](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN.py#L62)

#### Signature

```python
def forward(self, x, positions, edge_index, edge_feat=None): ...
```