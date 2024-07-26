# Norm Basis

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Allegro](../index.md#allegro) / [Nn](./index.md#nn) / Norm Basis

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._norm_basis](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_norm_basis.py) module.

- [Norm Basis](#norm-basis)
  - [NormalizedBasis](#normalizedbasis)
    - [NormalizedBasis().forward](#normalizedbasis()forward)

## NormalizedBasis

[Show source in _norm_basis.py:6](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_norm_basis.py#L6)

Normalized version of a given radial basis.

#### Arguments

- `basis` *constructor* - callable to build the underlying basis
- `basis_kwargs` *dict* - parameters for the underlying basis
- `n` *int, optional* - the number of samples to use for the estimated statistics
- `r_min` *float* - the lower bound of the uniform square bump distribution for inputs
- `r_max` *float* - the upper bound of the same

#### Signature

```python
class NormalizedBasis(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        r_min: float = 0.0,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        n: int = 4000,
        norm_basis_mean_shift: bool = True,
    ): ...
```

### NormalizedBasis().forward

[Show source in _norm_basis.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_norm_basis.py#L55)

#### Signature

```python
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```