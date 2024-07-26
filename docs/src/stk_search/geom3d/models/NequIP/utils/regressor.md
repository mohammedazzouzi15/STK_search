# Regressor

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Regressor

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils.regressor](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py) module.

- [Regressor](#regressor)
  - [NormalizedDotProduct](#normalizeddotproduct)
    - [NormalizedDotProduct().__call__](#normalizeddotproduct()__call__)
    - [NormalizedDotProduct().diag](#normalizeddotproduct()diag)
    - [NormalizedDotProduct().hyperparameter_diagonal_elements](#normalizeddotproduct()hyperparameter_diagonal_elements)
    - [NormalizedDotProduct().is_stationary](#normalizeddotproduct()is_stationary)
  - [base_gp](#base_gp)
  - [gp](#gp)
  - [normalized_gp](#normalized_gp)
  - [solver](#solver)

## NormalizedDotProduct

[Show source in regressor.py:109](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L109)

Dot-Product kernel.


```python
k(x_i, x_j) = x_i \cdot A \cdot x_j
```

#### Signature

```python
class NormalizedDotProduct(Kernel):
    def __init__(self, diagonal_elements): ...
```

### NormalizedDotProduct().__call__

[Show source in regressor.py:120](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L120)

Return the kernel k(X, Y) and optionally its gradient.
Parameters
----------
X : ndarray of shape (n_samples_X, n_features)
    Left argument of the returned kernel k(X, Y)
Y : ndarray of shape (n_samples_Y, n_features), default=None
    Right argument of the returned kernel k(X, Y). If None, k(X, X)
    if evaluated instead.
eval_gradient : bool, default=False
    Determines whether the gradient with respect to the log of
    the kernel hyperparameter is computed.
    Only supported when Y is None.
Returns
-------
K : ndarray of shape (n_samples_X, n_samples_Y)
    Kernel k(X, Y)
K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),                optional
    The gradient of the kernel k(X, X) with respect to the log of the
    hyperparameter of the kernel. Only returned when `eval_gradient`
    is True.

#### Signature

```python
def __call__(self, X, Y=None, eval_gradient=False): ...
```

### NormalizedDotProduct().diag

[Show source in regressor.py:156](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L156)

Returns the diagonal of the kernel k(X, X).
The result of this method is identical to np.diag(self(X)); however,
it can be evaluated more efficiently since only the diagonal is
evaluated.
Parameters
----------
X : ndarray of shape (n_samples_X, n_features)
    Left argument of the returned kernel k(X, Y).
Returns
-------
K_diag : ndarray of shape (n_samples_X,)
    Diagonal of kernel k(X, X).

#### Signature

```python
def diag(self, X): ...
```

### NormalizedDotProduct().hyperparameter_diagonal_elements

[Show source in regressor.py:179](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L179)

#### Signature

```python
@property
def hyperparameter_diagonal_elements(self): ...
```

### NormalizedDotProduct().is_stationary

[Show source in regressor.py:175](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L175)

Returns whether the kernel is stationary.

#### Signature

```python
def is_stationary(self): ...
```



## base_gp

[Show source in regressor.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L38)

#### Signature

```python
def base_gp(
    X,
    y,
    kernel,
    kernel_kwargs,
    alpha: Optional[float] = 0.1,
    max_iteration: int = 20,
    stride: Optional[int] = None,
): ...
```



## gp

[Show source in regressor.py:32](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L32)

#### Signature

```python
def gp(X, y, **kwargs): ...
```



## normalized_gp

[Show source in regressor.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L18)

#### Signature

```python
def normalized_gp(X, y, **kwargs): ...
```



## solver

[Show source in regressor.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/regressor.py#L9)

#### Signature

```python
def solver(X, y, regressor: Optional[str] = "NormalizedGaussianProcess", **kwargs): ...
```