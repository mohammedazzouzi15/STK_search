# So3

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [From Se3cnn](./index.md#from-se3cnn) / So3

> Auto-generated documentation for [src.stk_search.geom3d.models.from_se3cnn.SO3](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py) module.

- [So3](#so3)
  - [torch_default_dtype](#torch_default_dtype)
  - [_test_spherical_harmonics](#_test_spherical_harmonics)
  - [compose](#compose)
  - [irr_repr](#irr_repr)
  - [kron](#kron)
  - [rot](#rot)
  - [rot_y](#rot_y)
  - [rot_z](#rot_z)
  - [tensor3x3_repr](#tensor3x3_repr)
  - [tensor3x3_repr_basis_to_spherical_basis](#tensor3x3_repr_basis_to_spherical_basis)
  - [test_is_representation](#test_is_representation)
  - [x_to_alpha_beta](#x_to_alpha_beta)
  - [xyz_vector_basis_to_spherical_basis](#xyz_vector_basis_to_spherical_basis)

## torch_default_dtype

[Show source in SO3.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L13)

#### Signature

```python
class torch_default_dtype:
    def __init__(self, dtype): ...
```



## _test_spherical_harmonics

[Show source in SO3.py:259](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L259)

This test tests that
- irr_repr
- compose
- spherical_harmonics
are compatible

Y(Z(alpha) Y(beta) Z(gamma) x) = D(alpha, beta, gamma) Y(x)
with x = Z(a) Y(b) eta

#### Signature

```python
def _test_spherical_harmonics(order): ...
```



## compose

[Show source in SO3.py:132](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L132)

(a, b, c) = (a1, b1, c1) composed with (a2, b2, c2)

#### Signature

```python
def compose(a1, b1, c1, a2, b2, c2): ...
```



## irr_repr

[Show source in SO3.py:83](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L83)

irreducible representation of SO3
- compatible with compose and spherical_harmonics

#### Signature

```python
def irr_repr(order, alpha, beta, gamma, dtype=None): ...
```



## kron

[Show source in SO3.py:144](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L144)

#### Signature

```python
def kron(x, y): ...
```



## rot

[Show source in SO3.py:58](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L58)

ZYZ Eurler angles rotation

#### Signature

```python
def rot(alpha, beta, gamma): ...
```



## rot_y

[Show source in SO3.py:42](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L42)

Rotation around Y axis

#### Signature

```python
def rot_y(beta): ...
```



## rot_z

[Show source in SO3.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L26)

Rotation around Z axis

#### Signature

```python
def rot_z(gamma): ...
```



## tensor3x3_repr

[Show source in SO3.py:172](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L172)

representation of 3x3 tensors
T --> R T R^t

#### Signature

```python
def tensor3x3_repr(a, b, c): ...
```



## tensor3x3_repr_basis_to_spherical_basis

[Show source in SO3.py:181](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L181)

to convert a 3x3 tensor transforming with tensor3x3_repr(a, b, c)
into its 1 + 3 + 5 component transforming with irr_repr(0, a, b, c), irr_repr(1, a, b, c), irr_repr(3, a, b, c)
see assert for usage

#### Signature

```python
def tensor3x3_repr_basis_to_spherical_basis(): ...
```



## test_is_representation

[Show source in SO3.py:239](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L239)

rep(Z(a1) Y(b1) Z(c1) Z(a2) Y(b2) Z(c2)) = rep(Z(a1) Y(b1) Z(c1)) rep(Z(a2) Y(b2) Z(c2))

#### Signature

```python
def test_is_representation(rep): ...
```



## x_to_alpha_beta

[Show source in SO3.py:65](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L65)

Convert point (x, y, z) on the sphere into (alpha, beta)

#### Signature

```python
def x_to_alpha_beta(x): ...
```



## xyz_vector_basis_to_spherical_basis

[Show source in SO3.py:157](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/SO3.py#L157)

to convert a vector [x, y, z] transforming with rot(a, b, c)
into a vector transforming with irr_repr(1, a, b, c)
see assert for usage

#### Signature

```python
def xyz_vector_basis_to_spherical_basis(): ...
```