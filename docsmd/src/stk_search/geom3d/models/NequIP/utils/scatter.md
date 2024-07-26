# Scatter

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Scatter

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils.scatter](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/scatter.py) module.

- [Scatter](#scatter)
  - [scatter](#scatter)

## scatter

[Show source in scatter.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/scatter.py#L18)

#### Signature

```python
@torch.jit.script
def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor: ...
```