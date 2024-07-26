# Layout

[stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Allegro](../../index.md#allegro) / [Nn](../index.md#nn) / [Strided](./index.md#strided) / Layout

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._strided._layout](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_layout.py) module.

- [Layout](#layout)
  - [StridedLayout](#stridedlayout)
    - [StridedLayout.can_be_strided](#stridedlayoutcan_be_strided)
    - [StridedLayout().to_catted](#stridedlayout()to_catted)
    - [StridedLayout().to_strided](#stridedlayout()to_strided)

## StridedLayout

[Show source in _layout.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_layout.py#L10)

Utility class to represent a strided layout of a tensor whose irreps all have the same multiplicity.

#### Signature

```python
class StridedLayout:
    def __init__(self, irreps: Irreps, pad_to_multiple: int = 1): ...
```

### StridedLayout.can_be_strided

[Show source in _layout.py:62](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_layout.py#L62)

Check whether [irreps](#stridedlayout) is compatible with strided layout.

#### Signature

```python
@staticmethod
def can_be_strided(irreps: Irreps) -> bool: ...
```

### StridedLayout().to_catted

[Show source in _layout.py:74](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_layout.py#L74)

Convert a tensor from strided to default layout.

#### Signature

```python
def to_catted(self, x: torch.Tensor) -> torch.Tensor: ...
```

### StridedLayout().to_strided

[Show source in _layout.py:70](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_layout.py#L70)

Convert a tensor from default to strided layout.

#### Signature

```python
def to_strided(self, x: torch.Tensor) -> torch.Tensor: ...
```