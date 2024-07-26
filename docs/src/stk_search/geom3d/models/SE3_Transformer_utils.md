# Se3 Transformer Utils

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Se3 Transformer Utils

> Auto-generated documentation for [src.stk_search.geom3d.models.SE3_Transformer_utils](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py) module.

- [Se3 Transformer Utils](#se3-transformer-utils)
  - [G1x1SE3](#g1x1se3)
    - [G1x1SE3().forward](#g1x1se3()forward)
  - [GAttentiveSelfInt](#gattentiveselfint)
    - [GAttentiveSelfInt().forward](#gattentiveselfint()forward)
  - [GCat](#gcat)
    - [GCat().forward](#gcat()forward)
  - [GConvSE3Partial](#gconvse3partial)
    - [GConvSE3Partial().forward](#gconvse3partial()forward)
  - [GMABSE3](#gmabse3)
    - [GMABSE3().forward](#gmabse3()forward)
  - [GSE3Res](#gse3res)
    - [GSE3Res().forward](#gse3res()forward)
  - [GSum](#gsum)
    - [GSum().forward](#gsum()forward)

## G1x1SE3

[Show source in SE3_Transformer_utils.py:203](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L203)

Graph Linear SE(3)-equivariant layer, equivalent to a 1x1 convolution.
This is equivalent to a self-interaction layer in TensorField Networks.

#### Signature

```python
class G1x1SE3(nn.Module):
    def __init__(self, f_in, f_out, learnable=True): ...
```

### G1x1SE3().forward

[Show source in SE3_Transformer_utils.py:226](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L226)

#### Signature

```python
def forward(self, features, **kwargs): ...
```



## GAttentiveSelfInt

[Show source in SE3_Transformer_utils.py:307](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L307)

#### Signature

```python
class GAttentiveSelfInt(nn.Module):
    def __init__(self, f_in, f_out): ...
```

### GAttentiveSelfInt().forward

[Show source in SE3_Transformer_utils.py:340](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L340)

#### Signature

```python
def forward(self, features, **kwargs): ...
```



## GCat

[Show source in SE3_Transformer_utils.py:419](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L419)

Concat only degrees which are in f_x

#### Signature

```python
class GCat(nn.Module):
    def __init__(self, f_x: Fiber, f_y: Fiber): ...
```

### GCat().forward

[Show source in SE3_Transformer_utils.py:433](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L433)

#### Signature

```python
def forward(self, x, y): ...
```



## GConvSE3Partial

[Show source in SE3_Transformer_utils.py:97](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L97)

Graph SE(3)-equivariant node -> edge layer

#### Signature

```python
class GConvSE3Partial(nn.Module):
    def __init__(self, f_in, f_out, edge_dim: int = 0, x_ij=None): ...
```

### GConvSE3Partial().forward

[Show source in SE3_Transformer_utils.py:132](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L132)

Forward pass of the linear layer

#### Arguments

- `h` - dict of node-features
- `G` - minibatch of (homo)graphs
- `r` - inter-atomic distances
- `basis` - pre-computed Q * Y

#### Returns

tensor with new features [B, n_points, n_features_out]

#### Signature

```python
def forward(self, h, positions, r, basis, edge_index, edge_feat=None): ...
```



## GMABSE3

[Show source in SE3_Transformer_utils.py:237](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L237)

An SE(3)-equivariant multi-headed self-attention module for DGL graphs.

#### Signature

```python
class GMABSE3(nn.Module):
    def __init__(self, f_value, f_key, n_heads): ...
```

### GMABSE3().forward

[Show source in SE3_Transformer_utils.py:252](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L252)

Forward pass of the linear layer

#### Arguments

- `G` - minibatch of (homo)graphs
- `v` - dict of value edge-features
- `k` - dict of key edge-features
- `q` - dict of query node-features

#### Returns

tensor with new features [B, n_points, n_features_out]

#### Signature

```python
def forward(self, v, k, q, edge_index, **kwargs): ...
```



## GSE3Res

[Show source in SE3_Transformer_utils.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L10)

Graph attention block with SE(3)-equivariance and skip connection

#### Signature

```python
class GSE3Res(nn.Module):
    def __init__(
        self,
        f_in,
        f_out,
        edge_dim=0,
        div=4,
        n_heads=1,
        learnable_skip=True,
        skip="cat",
        selfint="1x1",
        x_ij=None,
    ): ...
```

### GSE3Res().forward

[Show source in SE3_Transformer_utils.py:78](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L78)

#### Signature

```python
def forward(self, h, edge_index, **kwargs): ...
```



## GSum

[Show source in SE3_Transformer_utils.py:375](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L375)

SE(3)-equvariant graph residual sum function.

#### Signature

```python
class GSum(nn.Module):
    def __init__(self, f_x: Fiber, f_y: Fiber): ...
```

### GSum().forward

[Show source in SE3_Transformer_utils.py:393](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SE3_Transformer_utils.py#L393)

#### Signature

```python
def forward(self, x, y): ...
```