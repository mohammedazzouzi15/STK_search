# Radial Basis

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Radial Basis

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn.radial_basis](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/radial_basis.py) module.

- [Radial Basis](#radial-basis)
  - [BesselBasis](#besselbasis)
    - [BesselBasis().forward](#besselbasis()forward)
  - [e3nn_basis](#e3nn_basis)
    - [e3nn_basis().forward](#e3nn_basis()forward)

## BesselBasis

[Show source in radial_basis.py:46](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/radial_basis.py#L46)

#### Signature

```python
class BesselBasis(nn.Module):
    def __init__(self, r_max, num_basis=8, trainable=True): ...
```

### BesselBasis().forward

[Show source in radial_basis.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/radial_basis.py#L81)

Evaluate Bessel Basis for input x.

Parameters
----------
x : torch.Tensor
    Input

#### Signature

```python
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```



## e3nn_basis

[Show source in radial_basis.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/radial_basis.py#L13)

#### Signature

```python
class e3nn_basis(nn.Module):
    def __init__(
        self,
        r_max: float,
        r_min: Optional[float] = None,
        e3nn_basis_name: str = "gaussian",
        num_basis: int = 8,
    ): ...
```

### e3nn_basis().forward

[Show source in radial_basis.py:32](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/radial_basis.py#L32)

#### Signature

```python
def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```