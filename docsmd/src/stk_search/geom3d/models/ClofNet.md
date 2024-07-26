# Clofnet

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Clofnet

> Auto-generated documentation for [src.stk_search.geom3d.models.ClofNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py) module.

- [Clofnet](#clofnet)
  - [CFConvS2V](#cfconvs2v)
    - [CFConvS2V().forward](#cfconvs2v()forward)
    - [CFConvS2V().message](#cfconvs2v()message)
  - [ClofNet](#clofnet)
    - [ClofNet().forward](#clofnet()forward)
    - [ClofNet().global_add_pool](#clofnet()global_add_pool)
    - [ClofNet().reset_parameters](#clofnet()reset_parameters)
  - [NeighborEmb](#neighboremb)
    - [NeighborEmb().forward](#neighboremb()forward)
    - [NeighborEmb().message](#neighboremb()message)
  - [ResidualLayer](#residuallayer)
    - [ResidualLayer().forward](#residuallayer()forward)
    - [ResidualLayer().reset_parameters](#residuallayer()reset_parameters)
  - [TransformerConv](#transformerconv)
    - [TransformerConv().forward](#transformerconv()forward)
    - [TransformerConv().message](#transformerconv()message)
    - [TransformerConv().reset_parameters](#transformerconv()reset_parameters)
  - [emb](#emb)
    - [emb().forward](#emb()forward)
    - [emb().reset_parameters](#emb()reset_parameters)
  - [init](#init)
    - [init().forward](#init()forward)
    - [init().reset_parameters](#init()reset_parameters)
  - [rbf_emb](#rbf_emb)
    - [rbf_emb().forward](#rbf_emb()forward)
    - [rbf_emb().reset_parameters](#rbf_emb()reset_parameters)

## CFConvS2V

[Show source in ClofNet.py:327](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L327)

#### Signature

```python
class CFConvS2V(MessagePassing):
    def __init__(self, hid_dim: int, **kwargs): ...
```

### CFConvS2V().forward

[Show source in ClofNet.py:342](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L342)

s (B, N, hid_dim)
v (B, N, 3, hid_dim)
ea (B, N, N)
ef (B, N, N, ef_dim)
ev (B, N, N, 3)
v (BN, 3, 1)
emb (BN, hid_dim)

#### Signature

```python
def forward(self, s, v, edge_index, emb): ...
```

### CFConvS2V().message

[Show source in ClofNet.py:363](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L363)

#### Signature

```python
def message(self, x_j, norm): ...
```



## ClofNet

[Show source in ClofNet.py:150](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L150)

#### Signature

```python
class ClofNet(torch.nn.Module):
    def __init__(
        self,
        energy_and_force=False,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=64,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_radial=12,
        num_radial2=80,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        heads=1,
        act="swish",
        output_init="GlorotOrthogonal",
        use_node_features=True,
        **kwargs
    ): ...
```

### ClofNet().forward

[Show source in ClofNet.py:228](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L228)

#### Signature

```python
def forward(self, z, pos, batch): ...
```

### ClofNet().global_add_pool

[Show source in ClofNet.py:220](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L220)

#### Signature

```python
def global_add_pool(
    x: Tensor, batch: Optional[Tensor], size: Optional[int] = None
) -> Tensor: ...
```

### ClofNet().reset_parameters

[Show source in ClofNet.py:216](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L216)

#### Signature

```python
def reset_parameters(self): ...
```



## NeighborEmb

[Show source in ClofNet.py:305](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L305)

#### Signature

```python
class NeighborEmb(MessagePassing):
    def __init__(self, hid_dim: int, **kwargs): ...
```

### NeighborEmb().forward

[Show source in ClofNet.py:315](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L315)

#### Signature

```python
def forward(self, z, s, edge_index, embs): ...
```

### NeighborEmb().message

[Show source in ClofNet.py:323](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L323)

#### Signature

```python
def message(self, x_j, norm): ...
```



## ResidualLayer

[Show source in ClofNet.py:96](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L96)

#### Signature

```python
class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act="swish"): ...
```

### ResidualLayer().forward

[Show source in ClofNet.py:111](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L111)

#### Signature

```python
def forward(self, x): ...
```

### ResidualLayer().reset_parameters

[Show source in ClofNet.py:105](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L105)

#### Signature

```python
def reset_parameters(self): ...
```



## TransformerConv

[Show source in ClofNet.py:369](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L369)

#### Signature

```python
class TransformerConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 2,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs
    ): ...
```

### TransformerConv().forward

[Show source in ClofNet.py:442](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L442)

#### Signature

```python
def forward(
    self,
    x: Union[Tensor, PairTensor],
    edge_index: Adj,
    edge_attr: OptTensor = None,
    edgeweight: OptTensor = None,
    emb: OptTensor = None,
    return_attention_weights=None,
): ...
```

### TransformerConv().message

[Show source in ClofNet.py:484](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L484)

#### Signature

```python
def message(
    self,
    query_i: Tensor,
    key_j: Tensor,
    value_j: Tensor,
    edge_attr: OptTensor,
    edgeweight: OptTensor,
    emb: OptTensor,
    index: Tensor,
    ptr: OptTensor,
    size_i: Optional[int],
) -> Tensor: ...
```

### TransformerConv().reset_parameters

[Show source in ClofNet.py:432](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L432)

#### Signature

```python
def reset_parameters(self): ...
```



## emb

[Show source in ClofNet.py:71](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L71)

#### Signature

```python
class emb(torch.nn.Module):
    def __init__(self, num_radial, cutoff, envelope_exponent): ...
```

### emb().forward

[Show source in ClofNet.py:87](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L87)

#### Signature

```python
def forward(self, dist): ...
```

### emb().reset_parameters

[Show source in ClofNet.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L81)

#### Signature

```python
def reset_parameters(self): ...
```



## init

[Show source in ClofNet.py:115](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L115)

#### Signature

```python
class init(torch.nn.Module):
    def __init__(
        self, num_radial, hidden_channels, act="swish", use_node_features=True
    ): ...
```

### init().forward

[Show source in ClofNet.py:137](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L137)

#### Signature

```python
def forward(self, x, emb, i, j): ...
```

### init().reset_parameters

[Show source in ClofNet.py:130](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L130)

#### Signature

```python
def reset_parameters(self): ...
```



## rbf_emb

[Show source in ClofNet.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L31)

modified: delete cutoff with r

#### Signature

```python
class rbf_emb(nn.Module):
    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False, **kwargs): ...
```

### rbf_emb().forward

[Show source in ClofNet.py:59](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L59)

#### Signature

```python
def forward(self, dist): ...
```

### rbf_emb().reset_parameters

[Show source in ClofNet.py:54](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ClofNet.py#L54)

#### Signature

```python
def reset_parameters(self): ...
```