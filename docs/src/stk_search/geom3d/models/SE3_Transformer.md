# Se3 Transformer

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Se3 Transformer

> Auto-generated documentation for [src.stk_search.geom3d.models.SE3_Transformer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer.py) module.

- [Se3 Transformer](#se3-transformer)
  - [SE3Transformer](#se3transformer)
    - [SE3Transformer().forward](#se3transformer()forward)

## SE3Transformer

[Show source in SE3_Transformer.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer.py#L11)

SE(3) equivariant GCN with attention

#### Signature

```python
class SE3Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        atom_feature_size,
        num_channels,
        num_nlayers=1,
        num_degrees=4,
        edge_dim=4,
        div=4,
        n_heads=1,
    ): ...
```

### SE3Transformer().forward

[Show source in SE3_Transformer.py:67](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer.py#L67)

#### Signature

```python
def forward(self, x, positions, edge_index, edge_feat=None): ...
```