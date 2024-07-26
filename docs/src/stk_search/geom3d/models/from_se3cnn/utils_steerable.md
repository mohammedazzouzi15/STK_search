# Utils Steerable

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [From Se3cnn](./index.md#from-se3cnn) / Utils Steerable

> Auto-generated documentation for [src.stk_search.geom3d.models.from_se3cnn.utils_steerable](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py) module.

- [Utils Steerable](#utils-steerable)
  - [ScalarActivation3rdDim](#scalaractivation3rddim)
    - [ScalarActivation3rdDim().forward](#scalaractivation3rddim()forward)
  - [_basis_transformation_Q_J](#_basis_transformation_q_j)
  - [get_matrices_kernel](#get_matrices_kernel)
  - [get_matrix_kernel](#get_matrix_kernel)
  - [get_maximum_order_unary_only](#get_maximum_order_unary_only)
  - [get_maximum_order_with_pairwise](#get_maximum_order_with_pairwise)
  - [get_spherical_from_cartesian](#get_spherical_from_cartesian)
  - [get_spherical_from_cartesian_torch](#get_spherical_from_cartesian_torch)
  - [kron](#kron)
  - [precompute_sh](#precompute_sh)
  - [spherical_harmonics](#spherical_harmonics)
  - [test_coordinate_conversion](#test_coordinate_conversion)

## ScalarActivation3rdDim

[Show source in utils_steerable.py:325](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L325)

#### Signature

```python
class ScalarActivation3rdDim(torch.nn.Module):
    def __init__(self, n_dim, activation, bias=True): ...
```

### ScalarActivation3rdDim().forward

[Show source in utils_steerable.py:342](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L342)

#### Arguments

- `input` - [B, N, s]

#### Signature

```python
def forward(self, input): ...
```



## _basis_transformation_Q_J

[Show source in utils_steerable.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L38)

#### Arguments

- `J` - order of the spherical harmonics
- `order_in` - order of the input representation
- `order_out` - order of the output representation

#### Returns

one part of the Q^-1 matrix of the article

#### Signature

```python
@cached_dirpklgz("cache/trans_Q")
def _basis_transformation_Q_J(J, order_in, order_out, version=3): ...
```



## get_matrices_kernel

[Show source in utils_steerable.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L31)

Computes the commun kernel of all the As matrices

#### Signature

```python
def get_matrices_kernel(As, eps=1e-10): ...
```



## get_matrix_kernel

[Show source in utils_steerable.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L15)

Compute an orthonormal basis of the kernel (x_1, x_2, ...)
A x_i = 0
scalar_product(x_i, x_j) = delta_ij

#### Arguments

- `A` - matrix

#### Returns

matrix where each row is a basis vector of the kernel of A

#### Signature

```python
def get_matrix_kernel(A, eps=1e-10): ...
```



## get_maximum_order_unary_only

[Show source in utils_steerable.py:239](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L239)

determine what spherical harmonics we need to pre-compute. if we have the
unary term only, we need to compare all adjacent layers

the spherical harmonics function depends on J (irrep order) purely, which is dedfined by
order_irreps = list(range(abs(order_in - order_out), order_in + order_out + 1))
simplification: we only care about the maximum (in some circumstances that means we calculate a few lower
order spherical harmonics which we won't actually need)

#### Arguments

- `per_layer_orders_and_multiplicities` - nested list of lists of 2-tuples

#### Returns

integer indicating maximum order J

#### Signature

```python
def get_maximum_order_unary_only(per_layer_orders_and_multiplicities): ...
```



## get_maximum_order_with_pairwise

[Show source in utils_steerable.py:272](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L272)

determine what spherical harmonics we need to pre-compute. for pairwise
interactions, this will just be twice the maximum order

the spherical harmonics function depends on J (irrep order) purely, which is defined by
order_irreps = list(range(abs(order_in - order_out), order_in + order_out + 1))
simplification: we only care about the maximum (in some circumstances that means we calculate a few lower
order spherical harmonics which we won't actually need)

#### Arguments

- `per_layer_orders_and_multiplicities` - nested list of lists of 2-tuples

#### Returns

integer indicating maximum order J

#### Signature

```python
def get_maximum_order_with_pairwise(per_layer_orders_and_multiplicities): ...
```



## get_spherical_from_cartesian

[Show source in utils_steerable.py:146](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L146)

#### Signature

```python
def get_spherical_from_cartesian(cartesian): ...
```



## get_spherical_from_cartesian_torch

[Show source in utils_steerable.py:86](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L86)

#### Signature

```python
def get_spherical_from_cartesian_torch(cartesian, divide_radius_by=1.0): ...
```



## kron

[Show source in utils_steerable.py:223](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L223)

A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk

Kronecker product of matrices a and b with leading batch dimensions.
Batch dimensions are broadcast. The number of them mush
:type a: torch.Tensor
:type b: torch.Tensor

#### Returns

Type: *torch.Tensor*

#### Signature

```python
def kron(a, b): ...
```



## precompute_sh

[Show source in utils_steerable.py:298](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L298)

pre-comput spherical harmonics up to order max_J

#### Arguments

- `r_ij` - relative positions
- `max_J` - maximum order used in entire network

#### Returns

dict where each entry has shape [B,N,K,2J+1]

#### Signature

```python
def precompute_sh(r_ij, max_J): ...
```



## spherical_harmonics

[Show source in utils_steerable.py:208](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L208)

spherical harmonics
- compatible with irr_repr and compose

computation time: excecuting 1000 times with array length 1 took 0.29 seconds;
executing it once with array of length 1000 took 0.0022 seconds

#### Signature

```python
def spherical_harmonics(order, alpha, beta, dtype=None): ...
```



## test_coordinate_conversion

[Show source in utils_steerable.py:201](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/utils_steerable.py#L201)

#### Signature

```python
def test_coordinate_conversion(): ...
```