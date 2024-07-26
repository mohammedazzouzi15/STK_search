# Cutoffs

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Cutoffs

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn.cutoffs](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/cutoffs.py) module.

- [Cutoffs](#cutoffs)
  - [PolynomialCutoff](#polynomialcutoff)
    - [PolynomialCutoff().forward](#polynomialcutoff()forward)

## PolynomialCutoff

[Show source in cutoffs.py:16](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/cutoffs.py#L16)

#### Signature

```python
class PolynomialCutoff(torch.nn.Module):
    def __init__(self, r_max: float, p: float = 6): ...
```

### PolynomialCutoff().forward

[Show source in cutoffs.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/cutoffs.py#L37)

Evaluate cutoff function.

x: torch.Tensor, input distance

#### Signature

```python
def forward(self, x): ...
```