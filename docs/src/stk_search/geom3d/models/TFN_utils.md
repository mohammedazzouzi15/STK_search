# Tfn Utils

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Tfn Utils

> Auto-generated documentation for [src.stk_search.geom3d.models.TFN_utils](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py) module.

- [Tfn Utils](#tfn-utils)
  - [BN](#bn)
    - [BN().forward](#bn()forward)
  - [GConvSE3](#gconvse3)
    - [GConvSE3().forward](#gconvse3()forward)
  - [GNormSE3](#gnormse3)
    - [GNormSE3().forward](#gnormse3()forward)
  - [PairwiseConv](#pairwiseconv)
    - [PairwiseConv().forward](#pairwiseconv()forward)
  - [RadialFunc](#radialfunc)
    - [RadialFunc().forward](#radialfunc()forward)

## BN

[Show source in TFN_utils.py:6](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L6)

SE(3)-equvariant batch/layer normalization

#### Signature

```python
class BN(nn.Module):
    def __init__(self, m): ...
```

### BN().forward

[Show source in TFN_utils.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L18)

#### Signature

```python
def forward(self, x): ...
```



## GConvSE3

[Show source in TFN_utils.py:111](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L111)

A tensor field network layer as a DGL module.
GConvSE3 stands for a Graph Convolution SE(3)-equivariant layer. It is the
equivalent of a linear layer in an MLP, a conv layer in a CNN, or a graph
conv layer in a GCN.
At each node, the activations are split into different "feature types",
indexed by the SE(3) representation type: non-negative integers 0, 1, 2, ..

#### Signature

```python
class GConvSE3(nn.Module):
    def __init__(
        self, f_in, f_out, self_interaction=False, edge_dim=0, flavor="skip"
    ): ...
```

### GConvSE3().forward

[Show source in TFN_utils.py:160](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L160)

Forward pass of the linear layer

#### Arguments

- `G` - minibatch of (homo)graphs
- `h` - dict of features
- `r` - inter-atomic distances
- `basis` - pre-computed Q * Y

#### Returns

tensor with new features [B, n_points, n_features_out]

#### Signature

```python
def forward(self, h, r, basis, edge_index, edge_feat=None, **kwargs): ...
```



## GNormSE3

[Show source in TFN_utils.py:221](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L221)

Graph Norm-based SE(3)-equivariant nonlinearity.
Nonlinearities are important in SE(3) equivariant GCNs. They are also quite
expensive to compute, so it is convenient for them to share resources with
other layers, such as normalization. The general workflow is as follows:
> for feature type in features:
>    norm, phase <- feature
>    output = fnc(norm) * phase
where fnc: {R+}^m -> R^m is a learnable map from m norms to m scalars.

#### Signature

```python
class GNormSE3(nn.Module):
    def __init__(self, fiber, nonlin=nn.ReLU(inplace=True), num_layers: int = 0): ...
```

### GNormSE3().forward

[Show source in TFN_utils.py:265](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L265)

#### Signature

```python
def forward(self, features, **kwargs): ...
```



## PairwiseConv

[Show source in TFN_utils.py:63](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L63)

SE(3)-equivariant convolution between two single-type features

#### Signature

```python
class PairwiseConv(nn.Module):
    def __init__(
        self, degree_in: int, nc_in: int, degree_out: int, nc_out: int, edge_dim: int = 0
    ): ...
```

### PairwiseConv().forward

[Show source in TFN_utils.py:101](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L101)

#### Signature

```python
def forward(self, feat, basis): ...
```



## RadialFunc

[Show source in TFN_utils.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L22)

NN parameterized radial profile function.

#### Signature

```python
class RadialFunc(nn.Module):
    def __init__(self, num_freq, in_dim, out_dim, edge_dim: int = 0): ...
```

### RadialFunc().forward

[Show source in TFN_utils.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TFN_utils.py#L55)

#### Signature

```python
def forward(self, x): ...
```