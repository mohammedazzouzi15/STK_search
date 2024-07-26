# RadialBasis

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / RadialBasis

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.radial_basis](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py) module.

- [RadialBasis](#radialbasis)
  - [BernsteinBasis](#bernsteinbasis)
    - [BernsteinBasis().forward](#bernsteinbasis()forward)
  - [ExponentialEnvelope](#exponentialenvelope)
    - [ExponentialEnvelope().forward](#exponentialenvelope()forward)
  - [PolynomialEnvelope](#polynomialenvelope)
    - [PolynomialEnvelope().forward](#polynomialenvelope()forward)
  - [RadialBasis](#radialbasis-1)
    - [RadialBasis().forward](#radialbasis()forward)
  - [SphericalBesselBasis](#sphericalbesselbasis)
    - [SphericalBesselBasis().forward](#sphericalbesselbasis()forward)

## BernsteinBasis

[Show source in radial_basis.py:91](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L91)

Bernstein polynomial basis,
as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
and Nonlocal Effects
Parameters
----------
num_radial: int
    Controls maximum frequency.
pregamma_initial: float
    Initial value of exponential coefficient gamma.
    Default: gamma = 0.5 * a_0**-1 = 0.94486,
    inverse softplus -> pregamma = log e**gamma - 1 = 0.45264

#### Signature

```python
class BernsteinBasis(torch.nn.Module):
    def __init__(self, num_radial: int, pregamma_initial: float = 0.45264): ...
```

### BernsteinBasis().forward

[Show source in radial_basis.py:131](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L131)

#### Signature

```python
def forward(self, d_scaled): ...
```



## ExponentialEnvelope

[Show source in radial_basis.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L37)

Exponential envelope function that ensures a smooth cutoff,
as proposed in Unke, Chmiela, Gastegger, Schütt, Sauceda, Müller 2021.
SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
and Nonlocal Effects

#### Signature

```python
class ExponentialEnvelope(torch.nn.Module):
    def __init__(self): ...
```

### ExponentialEnvelope().forward

[Show source in radial_basis.py:48](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L48)

#### Signature

```python
def forward(self, d_scaled): ...
```



## PolynomialEnvelope

[Show source in radial_basis.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L10)

Polynomial envelope function that ensures a smooth cutoff.
Parameters
----------
    exponent: int
        Exponent of the envelope function.

#### Signature

```python
class PolynomialEnvelope(torch.nn.Module):
    def __init__(self, exponent): ...
```

### PolynomialEnvelope().forward

[Show source in radial_basis.py:27](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L27)

#### Signature

```python
def forward(self, d_scaled): ...
```



## RadialBasis

[Show source in radial_basis.py:139](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L139)

Parameters
----------
num_radial: int
    Controls maximum frequency.
cutoff: float
    Cutoff distance in Angstrom.
rbf: dict = {"name": "gaussian"}
    Basis function and its hyperparameters.
envelope: dict = {"name": "polynomial", "exponent": 5}
    Envelope function and its hyperparameters.

#### Signature

```python
class RadialBasis(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        cutoff: float,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
    ): ...
```

### RadialBasis().forward

[Show source in radial_basis.py:192](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L192)

#### Signature

```python
def forward(self, d): ...
```



## SphericalBesselBasis

[Show source in radial_basis.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L55)

1D spherical Bessel basis
Parameters
----------
num_radial: int
    Controls maximum frequency.
cutoff: float
    Cutoff distance in Angstrom.

#### Signature

```python
class SphericalBesselBasis(torch.nn.Module):
    def __init__(self, num_radial: int, cutoff: float): ...
```

### SphericalBesselBasis().forward

[Show source in radial_basis.py:83](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/radial_basis.py#L83)

#### Signature

```python
def forward(self, d_scaled): ...
```