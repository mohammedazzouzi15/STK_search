# Representations

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [From Se3cnn](./index.md#from-se3cnn) / Representations

> Auto-generated documentation for [src.stk_search.geom3d.models.from_se3cnn.representations](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py) module.

#### Attributes

- `y` - y = tesseral_harmonics(l, m, theta, phi): sph_har.get_element(l, m, cu_theta, cu_phi).type(torch.float32)


- [Representations](#representations)
  - [SphericalHarmonics](#sphericalharmonics)
    - [SphericalHarmonics().clear](#sphericalharmonics()clear)
    - [SphericalHarmonics().get](#sphericalharmonics()get)
    - [SphericalHarmonics().get_element](#sphericalharmonics()get_element)
    - [SphericalHarmonics().lpmv](#sphericalharmonics()lpmv)
    - [SphericalHarmonics().negative_lpmv](#sphericalharmonics()negative_lpmv)
  - [lpmv](#lpmv)
  - [pochhammer](#pochhammer)
  - [semifactorial](#semifactorial)
  - [tesseral_harmonics](#tesseral_harmonics)

## SphericalHarmonics

[Show source in representations.py:108](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L108)

#### Signature

```python
class SphericalHarmonics(object):
    def __init__(self): ...
```

### SphericalHarmonics().clear

[Show source in representations.py:112](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L112)

#### Signature

```python
def clear(self): ...
```

### SphericalHarmonics().get

[Show source in representations.py:193](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L193)

Tesseral harmonic with Condon-Shortley phase.

The Tesseral spherical harmonics are also known as the real spherical
harmonics.

#### Arguments

- `l` - int for degree
- [theta](#representations) - collatitude or polar angle
- [phi](#representations) - longitude or azimuth

#### Returns

tensor of shape [*theta.shape, 2*l+1]

#### Signature

```python
def get(self, l, theta, phi, refresh=True): ...
```

### SphericalHarmonics().get_element

[Show source in representations.py:165](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L165)

Tesseral spherical harmonic with Condon-Shortley phase.

The Tesseral spherical harmonics are also known as the real spherical
harmonics.

#### Arguments

- `l` - int for degree
- `m` - int for order, where -l <= m < l
- [theta](#representations) - collatitude or polar angle
- [phi](#representations) - longitude or azimuth

#### Returns

tensor of shape theta

#### Signature

```python
def get_element(self, l, m, theta, phi): ...
```

### SphericalHarmonics().lpmv

[Show source in representations.py:121](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L121)

Associated Legendre function including Condon-Shortley phase.

#### Arguments

- `m` - int order
- `l` - int degree
- `x` - float argument tensor

#### Returns

tensor of x-shape

#### Signature

```python
def lpmv(self, l, m, x): ...
```

### SphericalHarmonics().negative_lpmv

[Show source in representations.py:115](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L115)

Compute negative order coefficients

#### Signature

```python
def negative_lpmv(self, l, m, y): ...
```



## lpmv

[Show source in representations.py:40](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L40)

Associated Legendre function including Condon-Shortley phase.

#### Arguments

- `m` - int order
- `l` - int degree
- `x` - float argument tensor

#### Returns

tensor of x-shape

#### Signature

```python
def lpmv(l, m, x): ...
```



## pochhammer

[Show source in representations.py:24](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L24)

Compute the pochhammer symbol (x)_k.

(x)_k = x * (x+1) * (x+2) *...* (x+k-1)

#### Arguments

- `x` - positive int

#### Returns

float for (x)_k

#### Signature

```python
def pochhammer(x, k): ...
```



## semifactorial

[Show source in representations.py:8](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L8)

Compute the semifactorial function x!!.

x!! = x * (x-2) * (x-4) *...

#### Arguments

- `x` - positive int

#### Returns

float for x!!

#### Signature

```python
def semifactorial(x): ...
```



## tesseral_harmonics

[Show source in representations.py:79](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/representations.py#L79)

Tesseral spherical harmonic with Condon-Shortley phase.

The Tesseral spherical harmonics are also known as the real spherical
harmonics.

#### Arguments

- `l` - int for degree
- `m` - int for order, where -l <= m < l
- [theta](#representations) - collatitude or polar angle
- [phi](#representations) - longitude or azimuth

#### Returns

tensor of shape theta

#### Signature

```python
def tesseral_harmonics(l, m, theta=0.0, phi=0.0): ...
```