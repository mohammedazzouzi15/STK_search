# One Hot

[stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Nequip](../../index.md#nequip) / [Nn](../index.md#nn) / [Embedding](./index.md#embedding) / One Hot

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn.embedding._one_hot](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_one_hot.py) module.

- [One Hot](#one-hot)
  - [OneHotAtomEncoding](#onehotatomencoding)
    - [OneHotAtomEncoding().forward](#onehotatomencoding()forward)

## OneHotAtomEncoding

[Show source in _one_hot.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_one_hot.py#L12)

Copmute a one-hot floating point encoding of atoms' discrete atom types.

#### Arguments

- `set_features` - If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.

#### Signature

```python
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    def __init__(self, num_types: int, set_features: bool = True, irreps_in=None): ...
```

### OneHotAtomEncoding().forward

[Show source in _one_hot.py:39](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_one_hot.py#L39)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```