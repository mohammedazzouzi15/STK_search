# DropPath

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Transformerm](../index.md#transformerm) / [Modules](./index.md#modules) / DropPath

> Auto-generated documentation for [src.stk_search.geom3d.models.TransformerM.modules.droppath](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/modules/droppath.py) module.

- [DropPath](#droppath)
  - [DropPath](#droppath-1)
    - [DropPath().extra_repr](#droppath()extra_repr)
    - [DropPath().forward](#droppath()forward)

## DropPath

[Show source in droppath.py:4](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/modules/droppath.py#L4)

Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

#### Signature

```python
class DropPath(torch.nn.Module):
    def __init__(self, prob=None): ...
```

### DropPath().extra_repr

[Show source in droppath.py:23](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/modules/droppath.py#L23)

#### Signature

```python
def extra_repr(self) -> str: ...
```

### DropPath().forward

[Show source in droppath.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/modules/droppath.py#L11)

#### Signature

```python
def forward(self, x): ...
```