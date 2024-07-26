# Spherenet Utils

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Spherenet Utils

> Auto-generated documentation for [src.stk_search.geom3d.models.SphereNet_utils](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py) module.

- [Spherenet Utils](#spherenet-utils)
  - [angle_emb](#angle_emb)
    - [angle_emb().forward](#angle_emb()forward)
  - [dist_emb](#dist_emb)
    - [dist_emb().forward](#dist_emb()forward)
    - [dist_emb().reset_parameters](#dist_emb()reset_parameters)
  - [torsion_emb](#torsion_emb)
    - [torsion_emb().forward](#torsion_emb()forward)
  - [real_sph_harm](#real_sph_harm)

## angle_emb

[Show source in SphereNet_utils.py:89](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L89)

#### Signature

```python
class angle_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5): ...
```

### angle_emb().forward

[Show source in SphereNet_utils.py:116](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L116)

#### Signature

```python
def forward(self, dist, angle, idx_kj): ...
```



## dist_emb

[Show source in SphereNet_utils.py:71](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L71)

#### Signature

```python
class dist_emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5): ...
```

### dist_emb().forward

[Show source in SphereNet_utils.py:84](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L84)

#### Signature

```python
def forward(self, dist): ...
```

### dist_emb().reset_parameters

[Show source in SphereNet_utils.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L81)

#### Signature

```python
def reset_parameters(self): ...
```



## torsion_emb

[Show source in SphereNet_utils.py:128](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L128)

#### Signature

```python
class torsion_emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5): ...
```

### torsion_emb().forward

[Show source in SphereNet_utils.py:160](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L160)

#### Signature

```python
def forward(self, dist, angle, phi, idx_kj): ...
```



## real_sph_harm

[Show source in SphereNet_utils.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet_utils.py#L11)

Computes formula strings of the the real part of the spherical harmonics up to order l (excluded).
Variables are either cartesian coordinates x,y,z on the unit sphere or spherical coordinates phi and theta.

#### Signature

```python
def real_sph_harm(l, zero_m_only=False, spherical_coordinates=True): ...
```