# Concat

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Concat

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._concat](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_concat.py) module.

- [Concat](#concat)
  - [Concat](#concat-1)
    - [Concat().forward](#concat()forward)

## Concat

[Show source in _concat.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_concat.py#L11)

Concatenate multiple fields into one.

#### Signature

```python
class Concat(GraphModuleMixin, torch.nn.Module):
    def __init__(self, in_fields: List[str], out_field: str, irreps_in={}): ...
```

### Concat().forward

[Show source in _concat.py:23](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_concat.py#L23)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```