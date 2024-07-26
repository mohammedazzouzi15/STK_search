# Painn Utils

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Painn Utils

> Auto-generated documentation for [src.stk_search.geom3d.models.PaiNN_utils](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py) module.

- [Painn Utils](#painn-utils)
  - [CosineCutoff](#cosinecutoff)
    - [CosineCutoff().forward](#cosinecutoff()forward)
  - [Dense](#dense)
    - [Dense().forward](#dense()forward)
    - [Dense().reset_parameters](#dense()reset_parameters)
  - [GaussianRBF](#gaussianrbf)
    - [GaussianRBF().forward](#gaussianrbf()forward)
  - [build_mlp](#build_mlp)
  - [cosine_cutoff](#cosine_cutoff)
  - [gaussian_rbf](#gaussian_rbf)
  - [replicate_module](#replicate_module)
  - [scatter_add](#scatter_add)

## CosineCutoff

[Show source in PaiNN_utils.py:160](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L160)

Behler-style cosine cutoff module.


```python
f(r) = \begin{cases}
 0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
   & r < r_\text{cutoff} \\
 0 & r \geqslant r_\text{cutoff} \\
 \end{cases}
```

#### Signature

```python
class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float): ...
```

### CosineCutoff().forward

[Show source in PaiNN_utils.py:178](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L178)

#### Signature

```python
def forward(self, input: torch.Tensor): ...
```



## Dense

[Show source in PaiNN_utils.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L9)

#### Signature

```python
class Dense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ): ...
```

### Dense().forward

[Show source in PaiNN_utils.py:32](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L32)

#### Signature

```python
def forward(self, input: torch.Tensor): ...
```

### Dense().reset_parameters

[Show source in PaiNN_utils.py:27](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L27)

#### Signature

```python
def reset_parameters(self): ...
```



## GaussianRBF

[Show source in PaiNN_utils.py:99](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L99)

Gaussian radial basis functions.

#### Signature

```python
class GaussianRBF(nn.Module):
    def __init__(
        self,
        n_rbf: int,
        cutoff: float,
        start: float = 0.0,
        trainable: bool = False,
        gamma: float = None,
    ): ...
```

### GaussianRBF().forward

[Show source in PaiNN_utils.py:127](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L127)

#### Signature

```python
def forward(self, inputs: torch.Tensor): ...
```



## build_mlp

[Show source in PaiNN_utils.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L38)

#### Signature

```python
def build_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
) -> nn.Module: ...
```



## cosine_cutoff

[Show source in PaiNN_utils.py:141](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L141)

 Behler-style cosine cutoff.
.. math

```python
f(r) = egin{cases}
 0.5 	imes \left[1 + \cos\left(
rac{\pi r}{r_	ext{cutoff}}
ight)
ight]
   & r < r_	ext{cutoff} \
 0 & r \geqslant r_	ext{cutoff} \
 \end{cases}
```

#### Arguments

- `cutoff` *float, optional* - cutoff radius.

#### Signature

```python
def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor): ...
```



## gaussian_rbf

[Show source in PaiNN_utils.py:135](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L135)

#### Signature

```python
def gaussian_rbf(coeff, inputs, offsets): ...
```



## replicate_module

[Show source in PaiNN_utils.py:89](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L89)

#### Signature

```python
def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
): ...
```



## scatter_add

[Show source in PaiNN_utils.py:73](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN_utils.py#L73)

#### Signature

```python
def scatter_add(
    x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0
) -> torch.Tensor: ...
```