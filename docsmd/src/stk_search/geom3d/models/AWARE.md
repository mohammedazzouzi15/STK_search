# Aware

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Aware

> Auto-generated documentation for [src.stk_search.geom3d.models.AWARE](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AWARE.py) module.

- [Aware](#aware)
  - [AWARE](#aware)
    - [AWARE().forward](#aware()forward)

## AWARE

[Show source in AWARE.py:6](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AWARE.py#L6)

#### Signature

```python
class AWARE(nn.Module):
    def __init__(
        self, emb_dim, r_prime, max_walk_len, num_layers, out_dim, use_bond=False
    ): ...
```

### AWARE().forward

[Show source in AWARE.py:43](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AWARE.py#L43)

#### Signature

```python
def forward(self, node_attribute_matrix, adjacent_matrix, adj_attr_matrix): ...
```