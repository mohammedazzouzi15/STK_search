# Util

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Util

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._util](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_util.py) module.

- [Util](#util)
  - [SaveForOutput](#saveforoutput)
    - [SaveForOutput().forward](#saveforoutput()forward)

## SaveForOutput

[Show source in _util.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_util.py#L7)

Copy a field and disconnect it from the autograd graph.

Copy a field and disconnect it from the autograd graph, storing it under another key for inspection as part of the models output.

#### Arguments

- `field` - the field to save
- `out_field` - the key to put the saved copy in

#### Signature

```python
class SaveForOutput(torch.nn.Module, GraphModuleMixin):
    def __init__(self, field: str, out_field: str, irreps_in=None): ...
```

### SaveForOutput().forward

[Show source in _util.py:27](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_util.py#L27)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```