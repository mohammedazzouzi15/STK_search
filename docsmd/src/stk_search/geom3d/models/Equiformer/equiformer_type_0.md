# Equiformer Type 0

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Equiformer Type 0

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.equiformer_type_0](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py) module.

- [Equiformer Type 0](#equiformer-type-0)
  - [AttnHeads2Vec](#attnheads2vec)
    - [AttnHeads2Vec().forward](#attnheads2vec()forward)
  - [ConcatIrrepsTensor](#concatirrepstensor)
    - [ConcatIrrepsTensor().check_sorted](#concatirrepstensor()check_sorted)
    - [ConcatIrrepsTensor().forward](#concatirrepstensor()forward)
    - [ConcatIrrepsTensor().get_ir_index](#concatirrepstensor()get_ir_index)
    - [ConcatIrrepsTensor().get_irreps_dim](#concatirrepstensor()get_irreps_dim)
  - [EdgeDegreeEmbeddingNetwork](#edgedegreeembeddingnetwork)
    - [EdgeDegreeEmbeddingNetwork().forward](#edgedegreeembeddingnetwork()forward)
  - [EquiformerEnergy](#equiformerenergy)
    - [EquiformerEnergy().build_blocks](#equiformerenergy()build_blocks)
    - [EquiformerEnergy().forward](#equiformerenergy()forward)
    - [EquiformerEnergy().no_weight_decay](#equiformerenergy()no_weight_decay)
  - [FeedForwardNetwork](#feedforwardnetwork)
    - [FeedForwardNetwork().forward](#feedforwardnetwork()forward)
  - [FullyConnectedTensorProductRescaleNorm](#fullyconnectedtensorproductrescalenorm)
    - [FullyConnectedTensorProductRescaleNorm().forward](#fullyconnectedtensorproductrescalenorm()forward)
  - [FullyConnectedTensorProductRescaleNormSwishGate](#fullyconnectedtensorproductrescalenormswishgate)
    - [FullyConnectedTensorProductRescaleNormSwishGate().forward](#fullyconnectedtensorproductrescalenormswishgate()forward)
  - [FullyConnectedTensorProductRescaleSwishGate](#fullyconnectedtensorproductrescaleswishgate)
    - [FullyConnectedTensorProductRescaleSwishGate().forward](#fullyconnectedtensorproductrescaleswishgate()forward)
  - [GraphAttention](#graphattention)
    - [GraphAttention().extra_repr](#graphattention()extra_repr)
    - [GraphAttention().forward](#graphattention()forward)
  - [NodeEmbeddingNetwork](#nodeembeddingnetwork)
    - [NodeEmbeddingNetwork().forward](#nodeembeddingnetwork()forward)
  - [ScaledScatter](#scaledscatter)
    - [ScaledScatter().extra_repr](#scaledscatter()extra_repr)
    - [ScaledScatter().forward](#scaledscatter()forward)
  - [SeparableFCTP](#separablefctp)
    - [SeparableFCTP().forward](#separablefctp()forward)
  - [SmoothLeakyReLU](#smoothleakyrelu)
    - [SmoothLeakyReLU().extra_repr](#smoothleakyrelu()extra_repr)
    - [SmoothLeakyReLU().forward](#smoothleakyrelu()forward)
  - [TransBlock](#transblock)
    - [TransBlock().forward](#transblock()forward)
  - [Vec2AttnHeads](#vec2attnheads)
    - [Vec2AttnHeads().forward](#vec2attnheads()forward)
  - [DepthwiseTensorProduct](#depthwisetensorproduct)
  - [Equiformer_l2](#equiformer_l2)
  - [Equiformer_nonlinear_bessel_l2](#equiformer_nonlinear_bessel_l2)
  - [Equiformer_nonlinear_bessel_l2_drop00](#equiformer_nonlinear_bessel_l2_drop00)
  - [Equiformer_nonlinear_bessel_l2_drop01](#equiformer_nonlinear_bessel_l2_drop01)
  - [Equiformer_nonlinear_l2](#equiformer_nonlinear_l2)
  - [Equiformer_nonlinear_l2_e3](#equiformer_nonlinear_l2_e3)
  - [get_mul_0](#get_mul_0)
  - [get_norm_layer](#get_norm_layer)

## AttnHeads2Vec

[Show source in equiformer_type_0.py:289](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L289)

Convert vectors of shape [N, num_heads, irreps_head] into
vectors of shape [N, irreps_head * num_heads].

#### Signature

```python
class AttnHeads2Vec(torch.nn.Module):
    def __init__(self, irreps_head): ...
```

### AttnHeads2Vec().forward

[Show source in equiformer_type_0.py:304](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L304)

#### Signature

```python
def forward(self, x): ...
```



## ConcatIrrepsTensor

[Show source in equiformer_type_0.py:319](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L319)

#### Signature

```python
class ConcatIrrepsTensor(torch.nn.Module):
    def __init__(self, irreps_1, irreps_2): ...
```

### ConcatIrrepsTensor().check_sorted

[Show source in equiformer_type_0.py:364](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L364)

#### Signature

```python
def check_sorted(self, irreps): ...
```

### ConcatIrrepsTensor().forward

[Show source in equiformer_type_0.py:384](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L384)

#### Signature

```python
def forward(self, feature_1, feature_2): ...
```

### ConcatIrrepsTensor().get_ir_index

[Show source in equiformer_type_0.py:377](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L377)

#### Signature

```python
def get_ir_index(self, ir, irreps): ...
```

### ConcatIrrepsTensor().get_irreps_dim

[Show source in equiformer_type_0.py:357](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L357)

#### Signature

```python
def get_irreps_dim(self, irreps): ...
```



## EdgeDegreeEmbeddingNetwork

[Show source in equiformer_type_0.py:709](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L709)

#### Signature

```python
class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(
        self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num
    ): ...
```

### EdgeDegreeEmbeddingNetwork().forward

[Show source in equiformer_type_0.py:725](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L725)

#### Signature

```python
def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch): ...
```



## EquiformerEnergy

[Show source in equiformer_type_0.py:736](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L736)

#### Signature

```python
class EquiformerEnergy(torch.nn.Module):
    def __init__(
        self,
        irreps_in="5x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        node_class=119,
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=[64, 64],
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1o+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="128x0e+64x1e+32x2e",
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        scale=None,
        atomref=None,
    ): ...
```

### EquiformerEnergy().build_blocks

[Show source in equiformer_type_0.py:810](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L810)

#### Signature

```python
def build_blocks(self): ...
```

### EquiformerEnergy().forward

[Show source in equiformer_type_0.py:865](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L865)

#### Signature

```python
def forward(self, pos, batch, node_atom, **kwargs) -> torch.Tensor: ...
```

### EquiformerEnergy().no_weight_decay

[Show source in equiformer_type_0.py:843](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L843)

#### Signature

```python
@torch.jit.ignore
def no_weight_decay(self): ...
```



## FeedForwardNetwork

[Show source in equiformer_type_0.py:537](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L537)

Use two (FCTP + Gate)

#### Signature

```python
class FeedForwardNetwork(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        irreps_mlp_mid=None,
        proj_drop=0.1,
    ): ...
```

### FeedForwardNetwork().forward

[Show source in equiformer_type_0.py:566](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L566)

#### Signature

```python
def forward(self, node_input, node_attr, **kwargs): ...
```



## FullyConnectedTensorProductRescaleNorm

[Show source in equiformer_type_0.py:78](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L78)

#### Signature

```python
class FullyConnectedTensorProductRescaleNorm(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer="graph",
    ): ...
```

### FullyConnectedTensorProductRescaleNorm().forward

[Show source in equiformer_type_0.py:92](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L92)

#### Signature

```python
def forward(self, x, y, batch, weight=None): ...
```



## FullyConnectedTensorProductRescaleNormSwishGate

[Show source in equiformer_type_0.py:98](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L98)

#### Signature

```python
class FullyConnectedTensorProductRescaleNormSwishGate(
    FullyConnectedTensorProductRescaleNorm
):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
        norm_layer="graph",
    ): ...
```

#### See also

- [FullyConnectedTensorProductRescaleNorm](#fullyconnectedtensorproductrescalenorm)

### FullyConnectedTensorProductRescaleNormSwishGate().forward

[Show source in equiformer_type_0.py:121](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L121)

#### Signature

```python
def forward(self, x, y, batch, weight=None): ...
```



## FullyConnectedTensorProductRescaleSwishGate

[Show source in equiformer_type_0.py:128](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L128)

#### Signature

```python
class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ): ...
```

### FullyConnectedTensorProductRescaleSwishGate().forward

[Show source in equiformer_type_0.py:151](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L151)

#### Signature

```python
def forward(self, x, y, weight=None): ...
```



## GraphAttention

[Show source in equiformer_type_0.py:403](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L403)

1. Message = Alpha * Value
2. Two Linear to merge src and dst -> Separable FCTP -> 0e + (0e+1e+...)
3. 0e -> Activation -> Inner Product -> (Alpha)
4. (0e+1e+...) -> (Value)

#### Signature

```python
class GraphAttention(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
    ): ...
```

### GraphAttention().extra_repr

[Show source in equiformer_type_0.py:530](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L530)

#### Signature

```python
def extra_repr(self): ...
```

### GraphAttention().forward

[Show source in equiformer_type_0.py:482](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L482)

#### Signature

```python
def forward(
    self,
    node_input,
    node_attr,
    edge_src,
    edge_dst,
    edge_attr,
    edge_scalars,
    batch,
    **kwargs
): ...
```



## NodeEmbeddingNetwork

[Show source in equiformer_type_0.py:670](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L670)

#### Signature

```python
class NodeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, max_atom_type, bias=True): ...
```

### NodeEmbeddingNetwork().forward

[Show source in equiformer_type_0.py:682](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L682)

`node_atom` is a LongTensor.

#### Signature

```python
def forward(self, node_atom): ...
```



## ScaledScatter

[Show source in equiformer_type_0.py:693](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L693)

#### Signature

```python
class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num): ...
```

### ScaledScatter().extra_repr

[Show source in equiformer_type_0.py:705](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L705)

#### Signature

```python
def extra_repr(self): ...
```

### ScaledScatter().forward

[Show source in equiformer_type_0.py:699](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L699)

#### Signature

```python
def forward(self, x, index, **kwargs): ...
```



## SeparableFCTP

[Show source in equiformer_type_0.py:186](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L186)

Use separable FCTP for spatial convolution.

#### Signature

```python
class SeparableFCTP(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        use_activation=False,
        norm_layer="graph",
        internal_weights=False,
    ): ...
```

### SeparableFCTP().forward

[Show source in equiformer_type_0.py:234](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L234)

Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
self.dtp_rad(`edge_scalars`).

#### Signature

```python
def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs): ...
```



## SmoothLeakyReLU

[Show source in equiformer_type_0.py:54](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L54)

#### Signature

```python
class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2): ...
```

### SmoothLeakyReLU().extra_repr

[Show source in equiformer_type_0.py:66](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L66)

#### Signature

```python
def extra_repr(self): ...
```

### SmoothLeakyReLU().forward

[Show source in equiformer_type_0.py:60](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L60)

#### Signature

```python
def forward(self, x): ...
```



## TransBlock

[Show source in equiformer_type_0.py:575](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L575)

1. Layer Norm 1 -> GraphAttention -> Layer Norm 2 -> FeedForwardNetwork
2. Use pre-norm architecture

#### Signature

```python
class TransBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        alpha_drop=0.1,
        proj_drop=0.1,
        drop_path_rate=0.0,
        irreps_mlp_mid=None,
        norm_layer="layer",
    ): ...
```

### TransBlock().forward

[Show source in equiformer_type_0.py:639](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L639)

#### Signature

```python
def forward(
    self,
    node_input,
    node_attr,
    edge_src,
    edge_dst,
    edge_attr,
    edge_scalars,
    batch,
    **kwargs
): ...
```



## Vec2AttnHeads

[Show source in equiformer_type_0.py:252](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L252)

Reshape vectors of shape [N, irreps_mid] to vectors of shape
[N, num_heads, irreps_head].

#### Signature

```python
class Vec2AttnHeads(torch.nn.Module):
    def __init__(self, irreps_head, num_heads): ...
```

### Vec2AttnHeads().forward

[Show source in equiformer_type_0.py:272](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L272)

#### Signature

```python
def forward(self, x): ...
```



## DepthwiseTensorProduct

[Show source in equiformer_type_0.py:157](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L157)

The irreps of output is pre-determined.
`irreps_node_output` is used to get certain types of vectors.

#### Signature

```python
def DepthwiseTensorProduct(
    irreps_node_input,
    irreps_edge_attr,
    irreps_node_output,
    internal_weights=False,
    bias=True,
): ...
```



## Equiformer_l2

[Show source in equiformer_type_0.py:903](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L903)

#### Signature

```python
def Equiformer_l2(irreps_in, radius, num_basis, node_class, **kwargs): ...
```



## Equiformer_nonlinear_bessel_l2

[Show source in equiformer_type_0.py:952](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L952)

#### Signature

```python
def Equiformer_nonlinear_bessel_l2(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_bessel_l2_drop00

[Show source in equiformer_type_0.py:986](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L986)

#### Signature

```python
def Equiformer_nonlinear_bessel_l2_drop00(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_bessel_l2_drop01

[Show source in equiformer_type_0.py:969](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L969)

#### Signature

```python
def Equiformer_nonlinear_bessel_l2_drop01(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_l2

[Show source in equiformer_type_0.py:919](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L919)

#### Signature

```python
def Equiformer_nonlinear_l2(irreps_in, radius, num_basis, node_class, **kwargs): ...
```



## Equiformer_nonlinear_l2_e3

[Show source in equiformer_type_0.py:935](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L935)

#### Signature

```python
def Equiformer_nonlinear_l2_e3(irreps_in, radius, num_basis, node_class, **kwargs): ...
```



## get_mul_0

[Show source in equiformer_type_0.py:70](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L70)

#### Signature

```python
def get_mul_0(irreps): ...
```



## get_norm_layer

[Show source in equiformer_type_0.py:39](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0.py#L39)

#### Signature

```python
def get_norm_layer(norm_type): ...
```