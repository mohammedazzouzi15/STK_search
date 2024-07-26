# Features

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Pronet](./index.md#pronet) / Features

> Auto-generated documentation for [src.stk_search.geom3d.models.ProNet.features](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py) module.

- [Features](#features)
  - [d_angle_emb](#d_angle_emb)
    - [d_angle_emb().forward](#d_angle_emb()forward)
  - [d_theta_phi_emb](#d_theta_phi_emb)
    - [d_theta_phi_emb().forward](#d_theta_phi_emb()forward)
  - [Jn](#jn)
  - [Jn_zeros](#jn_zeros)
  - [associated_legendre_polynomials](#associated_legendre_polynomials)
  - [bessel_basis](#bessel_basis)
  - [real_sph_harm](#real_sph_harm)
  - [sph_harm_prefactor](#sph_harm_prefactor)
  - [spherical_bessel_formulas](#spherical_bessel_formulas)

## d_angle_emb

[Show source in features.py:253](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L253)

#### Signature

```python
class d_angle_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0): ...
```

### d_angle_emb().forward

[Show source in features.py:285](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L285)

#### Signature

```python
def forward(self, dist, angle): ...
```



## d_theta_phi_emb

[Show source in features.py:294](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L294)

#### Signature

```python
class d_theta_phi_emb(torch.nn.Module):
    def __init__(self, num_radial, num_spherical, cutoff=8.0): ...
```

### d_theta_phi_emb().forward

[Show source in features.py:336](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L336)

#### Signature

```python
def forward(self, dist, theta, phi): ...
```



## Jn

[Show source in features.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L12)

numerical spherical bessel functions of order n

#### Signature

```python
def Jn(r, n): ...
```



## Jn_zeros

[Show source in features.py:19](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L19)

Compute the first k zeros of the spherical bessel functions up to order n (excluded)

#### Signature

```python
def Jn_zeros(n, k): ...
```



## associated_legendre_polynomials

[Show source in features.py:108](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L108)

Computes string formulas of the associated legendre polynomials up to degree L (excluded).
Parameters
----------
    L: int
        Degree up to which to calculate the associated legendre polynomials (degree L is excluded).
    zero_m_only: bool
        If True only calculate the polynomials for the polynomials where m=0.
    pos_m_only: bool
        If True only calculate the polynomials for the polynomials where m>=0. Overwritten by zero_m_only.
Returns
-------
    polynomials: list
        Contains the sympy functions of the polynomials (in total L many if zero_m_only is True else L^2 many).

#### Signature

```python
def associated_legendre_polynomials(L, zero_m_only=True, pos_m_only=True): ...
```



## bessel_basis

[Show source in features.py:52](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L52)

Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
order n (excluded) and maximum frequency k (excluded).

#### Returns

- `bess_basis` - list
    Bessel basis formulas taking in a single argument x.
    Has length n where each element has length k. -> In total n*k many.

#### Signature

```python
def bessel_basis(n, k): ...
```



## real_sph_harm

[Show source in features.py:173](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L173)

Computes formula strings of the the real part of the spherical harmonics up to degree L (excluded).
Variables are either spherical coordinates phi and theta (or cartesian coordinates x,y,z) on the UNIT SPHERE.
Parameters
----------
    L: int
        Degree up to which to calculate the spherical harmonics (degree L is excluded).
    spherical_coordinates: bool
        - True: Expects the input of the formula strings to be phi and theta.
        - False: Expects the input of the formula strings to be x, y and z.
    zero_m_only: bool
        If True only calculate the harmonics where m=0.
Returns
-------
    Y_lm_real: list
        Computes formula strings of the the real part of the spherical harmonics up
        to degree L (where degree L is not excluded).
        In total L^2 many sph harm exist up to degree L (excluded). However, if zero_m_only only is True then
        the total count is reduced to be only L many.

#### Signature

```python
def real_sph_harm(L, spherical_coordinates, zero_m_only=True): ...
```



## sph_harm_prefactor

[Show source in features.py:87](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L87)

Computes the constant pre-factor for the spherical harmonic of degree l and order m.
Parameters
----------
    l: int
        Degree of the spherical harmonic. l >= 0
    m: int
        Order of the spherical harmonic. -l <= m <= l
Returns
-------
    factor: float

#### Signature

```python
def sph_harm_prefactor(l, m): ...
```



## spherical_bessel_formulas

[Show source in features.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/features.py#L37)

Computes the sympy formulas for the spherical bessel functions up to order n (excluded)

#### Signature

```python
def spherical_bessel_formulas(n): ...
```