# Gvp

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Gvp

> Auto-generated documentation for [src.stk_search.geom3d.models.GVP](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py) module.

- [Gvp](#gvp)
  - [Dropout](#dropout)
    - [Dropout().forward](#dropout()forward)
  - [GVP](#gvp)
    - [GVP().forward](#gvp()forward)
  - [GVPConv](#gvpconv)
    - [GVPConv().forward](#gvpconv()forward)
    - [GVPConv().message](#gvpconv()message)
  - [GVPConvLayer](#gvpconvlayer)
    - [GVPConvLayer().forward](#gvpconvlayer()forward)
  - [GVP_GNN](#gvp_gnn)
    - [GVP_GNN().forward](#gvp_gnn()forward)
  - [LayerNorm](#layernorm)
    - [LayerNorm().forward](#layernorm()forward)
  - [_VDropout](#_vdropout)
    - [_VDropout().forward](#_vdropout()forward)
  - [_merge](#_merge)
  - [_norm_no_nan](#_norm_no_nan)
  - [_split](#_split)
  - [randn](#randn)
  - [tuple_cat](#tuple_cat)
  - [tuple_index](#tuple_index)
  - [tuple_sum](#tuple_sum)

## Dropout

[Show source in GVP.py:172](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L172)

Combined dropout for tuples (s, V).
Takes tuples (s, V) as input and as output.

#### Signature

```python
class Dropout(nn.Module):
    def __init__(self, drop_rate): ...
```

### Dropout().forward

[Show source in GVP.py:182](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L182)

#### Arguments

- `x` - tuple (s, V) of `torch.Tensor`,
          or single `torch.Tensor`
          (will be assumed to be scalar channels)

#### Signature

```python
def forward(self, x): ...
```



## GVP

[Show source in GVP.py:83](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L83)

Geometric Vector Perceptron. See manuscript and README.md
for more details.

#### Arguments

- `in_dims` - tuple (n_scalar, n_vector)
- `out_dims` - tuple (n_scalar, n_vector)
- `h_dim` - intermediate number of vector channels, optional
- `activations` - tuple of functions (scalar_act, vector_act)
- `vector_gate` - whether to use vector gating.
                    (vector_act will be used as sigma^+ in vector gating if `True`)

#### Signature

```python
class GVP(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ): ...
```

### GVP().forward

[Show source in GVP.py:114](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L114)

#### Arguments

- `x` - tuple (s, V) of `torch.Tensor`,
          or (if vectors_in is 0), a single `torch.Tensor`

#### Returns

tuple (s, V) of `torch.Tensor`,
         or (if vectors_out is 0), a single `torch.Tensor`

#### Signature

```python
def forward(self, x): ...
```



## GVPConv

[Show source in GVP.py:216](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L216)

Graph convolution / message passing with Geometric Vector Perceptrons.
Takes in a graph with node and edge embeddings,
and returns new node embeddings.

This does NOT do residual updates and pointwise feedforward layers
---see [GVPConvLayer](#gvpconvlayer).

#### Arguments

- `in_dims` - input node embedding dimensions (n_scalar, n_vector)
- `out_dims` - output node embedding dimensions (n_scalar, n_vector)
- `edge_dims` - input edge embedding dimensions (n_scalar, n_vector)
- `n_layers` - number of GVPs in the message function
- `module_list` - preconstructed message function, overrides n_layers
- `aggr` - should be "add" if some incoming edges are masked, as in
             a masked autoregressive decoder architecture, otherwise "mean"
- `activations` - tuple of functions (scalar_act, vector_act) to use in GVPs
- `vector_gate` - whether to use vector gating.
                    (vector_act will be used as sigma^+ in vector gating if `True`)

#### Signature

```python
class GVPConv(MessagePassing):
    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers=3,
        module_list=None,
        aggr="mean",
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ): ...
```

### GVPConv().forward

[Show source in GVP.py:263](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L263)

#### Arguments

- `x` - tuple (s, V) of `torch.Tensor`
- `edge_index` - array of shape [2, n_edges]
- `edge_attr` - tuple (s, V) of `torch.Tensor`

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GVPConv().message

[Show source in GVP.py:275](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L275)

#### Signature

```python
def message(self, s_i, v_i, s_j, v_j, edge_attr): ...
```



## GVPConvLayer

[Show source in GVP.py:283](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L283)

Full graph convolution / message passing layer with
Geometric Vector Perceptrons. Residually updates node embeddings with
aggregated incoming messages, applies a pointwise feedforward
network to node embeddings, and returns updated node embeddings.

To only compute the aggregated messages, see [GVPConv](#gvpconv).

#### Arguments

- `node_dims` - node embedding dimensions (n_scalar, n_vector)
- `edge_dims` - input edge embedding dimensions (n_scalar, n_vector)
- `n_message` - number of GVPs to use in message function
- `n_feedforward` - number of GVPs to use in feedforward function
- `drop_rate` - drop probability in all dropout layers
- `autoregressive` - if `True`, this [GVPConvLayer](#gvpconvlayer) will be used
       with a different set of input node embeddings for messages
       where src >= dst
- `activations` - tuple of functions (scalar_act, vector_act) to use in GVPs
- `vector_gate` - whether to use vector gating.
                    (vector_act will be used as sigma^+ in vector gating if `True`)

#### Signature

```python
class GVPConvLayer(nn.Module):
    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        autoregressive=False,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ): ...
```

### GVPConvLayer().forward

[Show source in GVP.py:329](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L329)

#### Arguments

- `x` - tuple (s, V) of `torch.Tensor`
- `edge_index` - array of shape [2, n_edges]
- `edge_attr` - tuple (s, V) of `torch.Tensor`
- `autoregressive_x` - tuple (s, V) of `torch.Tensor`.
        If not `None`, will be used as src node embeddings
        for forming messages where src >= dst. The corrent node
        embeddings `x` will still be the base of the update and the
        pointwise feedforward.
- `node_mask` - array of type `bool` to index into the first
        dim of node embeddings (s, V). If not `None`, only
        these nodes will be updated.

#### Signature

```python
def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None): ...
```



## GVP_GNN

[Show source in GVP.py:421](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L421)

A base 5-layer GVP-GNN for all ATOM3D tasks, using GVPs with
vector gating as described in the manuscript. Takes in atomic-level
structure graphs of type `torch_geometric.data.Batch`
and returns a single scalar.

This class should not be used directly. Instead, please use the
task-specific models which extend BaseModel. (Some of these classes
may be aliases of BaseModel.)

#### Arguments

- `num_rbf` - number of radial bases to use in the edge embedding

#### Signature

```python
class GVP_GNN(nn.Module):
    def __init__(self, num_rbf=16, out_channels=1195): ...
```

### GVP_GNN().forward

[Show source in GVP.py:471](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L471)

Forward pass which can be adjusted based on task formulation.

#### Arguments

- `batch` - `torch_geometric.data.Batch` with data attributes
              as returned from a BaseTransform
- `scatter_mean` - if `True`, returns mean of final node embeddings
                     (for each graph), else, returns embeddings seperately
- `dense` - if `True`, applies final dense layer to reduce embedding
              to a single scalar; else, returns the embedding

#### Signature

```python
def forward(self, batch, scatter_mean=True, dense=True): ...
```



## LayerNorm

[Show source in GVP.py:193](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L193)

Combined LayerNorm for tuples (s, V).
Takes tuples (s, V) as input and as output.

#### Signature

```python
class LayerNorm(nn.Module):
    def __init__(self, dims): ...
```

### LayerNorm().forward

[Show source in GVP.py:203](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L203)

#### Arguments

- `x` - tuple (s, V) of `torch.Tensor`,
          or single `torch.Tensor`
          (will be assumed to be scalar channels)

#### Signature

```python
def forward(self, x): ...
```



## _VDropout

[Show source in GVP.py:149](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L149)

Vector channel dropout where the elements of each
vector channel are dropped together.

#### Signature

```python
class _VDropout(nn.Module):
    def __init__(self, drop_rate): ...
```

### _VDropout().forward

[Show source in GVP.py:159](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L159)

#### Arguments

- `x` - `torch.Tensor` corresponding to vector channels

#### Signature

```python
def forward(self, x): ...
```



## _merge

[Show source in GVP.py:72](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L72)

Merges a tuple (s, V) into a single `torch.Tensor`, where the
vector channels are flattened and appended to the scalar channels.
Should be used only if the tuple representation cannot be used.
Use `_split(x, nv)` to reverse.

#### Signature

```python
def _merge(s, v): ...
```



## _norm_no_nan

[Show source in GVP.py:50](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L50)

L2 norm of tensor clamped above a minimum value `eps`.

#### Arguments

- `sqrt` - if `False`, returns the square of the L2 norm

#### Signature

```python
def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-08, sqrt=True): ...
```



## _split

[Show source in GVP.py:59](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L59)

Splits a merged representation of (s, V) back into a tuple.
Should be used only with `_merge(s, V)` and only if the tuple
representation cannot be used.

#### Arguments

- `x` - the `torch.Tensor` returned from `_merge`
- `nv` - the number of vector channels in the input to `_merge`

#### Signature

```python
def _split(x, nv): ...
```



## randn

[Show source in GVP.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L37)

Returns random tuples (s, V) drawn elementwise from a normal distribution.

#### Arguments

- `n` - number of data points
- `dims` - tuple of dimensions (n_scalar, n_vector)

#### Returns

(s, V) with s.shape = (n, n_scalar) and
         V.shape = (n, n_vector, 3)

#### Signature

```python
def randn(n, dims, device="cpu"): ...
```



## tuple_cat

[Show source in GVP.py:16](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L16)

Concatenates any number of tuples (s, V) elementwise.

#### Arguments

- `dim` - dimension along which to concatenate when viewed
            as the `dim` index for the scalar-channel tensors.
            This means that `dim=-1` will be applied as
            `dim=-2` for the vector-channel tensors.

#### Signature

```python
def tuple_cat(dim=-1, *args): ...
```



## tuple_index

[Show source in GVP.py:29](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L29)

Indexes into a tuple (s, V) along the first dimension.

#### Arguments

- `idx` - any object which can be used to index into a `torch.Tensor`

#### Signature

```python
def tuple_index(x, idx): ...
```



## tuple_sum

[Show source in GVP.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GVP.py#L10)

Sums any number of tuples (s, V) elementwise.

#### Signature

```python
def tuple_sum(*args): ...
```