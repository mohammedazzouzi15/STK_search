# Mlp

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Mlp

> Auto-generated documentation for [src.stk_search.geom3d.models.MLP](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/MLP.py) module.

- [Mlp](#mlp)
  - [MLP](#mlp)
    - [MLP().forward](#mlp()forward)
    - [MLP().represent](#mlp()represent)

## MLP

[Show source in MLP.py:5](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/MLP.py#L5)

#### Signature

```python
class MLP(nn.Module):
    def __init__(self, ECFP_dim, hidden_dim, output_dim): ...
```

### MLP().forward

[Show source in MLP.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/MLP.py#L26)

#### Signature

```python
def forward(self, x): ...
```

### MLP().represent

[Show source in MLP.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/MLP.py#L22)

#### Signature

```python
def represent(self, x): ...
```