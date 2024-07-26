# Equiformer Type 0 Periodic

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Equiformer Type 0 Periodic

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.equiformer_type_0_periodic](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py) module.

- [Equiformer Type 0 Periodic](#equiformer-type-0-periodic)
  - [AttnHeads2Vec](#attnheads2vec)
    - [AttnHeads2Vec().forward](#attnheads2vec()forward)
  - [ConcatIrrepsTensor](#concatirrepstensor)
    - [ConcatIrrepsTensor().check_sorted](#concatirrepstensor()check_sorted)
    - [ConcatIrrepsTensor().forward](#concatirrepstensor()forward)
    - [ConcatIrrepsTensor().get_ir_index](#concatirrepstensor()get_ir_index)
    - [ConcatIrrepsTensor().get_irreps_dim](#concatirrepstensor()get_irreps_dim)
  - [EdgeDegreeEmbeddingNetwork](#edgedegreeembeddingnetwork)
    - [EdgeDegreeEmbeddingNetwork().forward](#edgedegreeembeddingnetwork()forward)
  - [EquiformerEnergyPeriodic](#equiformerenergyperiodic)
    - [EquiformerEnergyPeriodic().build_blocks](#equiformerenergyperiodic()build_blocks)
    - [EquiformerEnergyPeriodic().forward](#equiformerenergyperiodic()forward)
    - [EquiformerEnergyPeriodic().forward_with_gathered_index](#equiformerenergyperiodic()forward_with_gathered_index)
    - [EquiformerEnergyPeriodic().no_weight_decay](#equiformerenergyperiodic()no_weight_decay)
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
  - [get_mul_0](#get_mul_0)
  - [get_norm_layer](#get_norm_layer)

## AttnHeads2Vec

[Show source in equiformer_type_0_periodic.py:290](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L290)

Convert vectors of shape [N, num_heads, irreps_head] into
vectors of shape [N, irreps_head * num_heads].

#### Signature

```python
class AttnHeads2Vec(torch.nn.Module):
    def __init__(self, irreps_head): ...
```

### AttnHeads2Vec().forward

[Show source in equiformer_type_0_periodic.py:305](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L305)

#### Signature

```python
def forward(self, x): ...
```



## ConcatIrrepsTensor

[Show source in equiformer_type_0_periodic.py:320](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L320)

#### Signature

```python
class ConcatIrrepsTensor(torch.nn.Module):
    def __init__(self, irreps_1, irreps_2): ...
```

### ConcatIrrepsTensor().check_sorted

[Show source in equiformer_type_0_periodic.py:365](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L365)

#### Signature

```python
def check_sorted(self, irreps): ...
```

### ConcatIrrepsTensor().forward

[Show source in equiformer_type_0_periodic.py:385](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L385)

#### Signature

```python
def forward(self, feature_1, feature_2): ...
```

### ConcatIrrepsTensor().get_ir_index

[Show source in equiformer_type_0_periodic.py:378](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L378)

#### Signature

```python
def get_ir_index(self, ir, irreps): ...
```

### ConcatIrrepsTensor().get_irreps_dim

[Show source in equiformer_type_0_periodic.py:358](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L358)

#### Signature

```python
def get_irreps_dim(self, irreps): ...
```



## EdgeDegreeEmbeddingNetwork

[Show source in equiformer_type_0_periodic.py:710](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L710)

#### Signature

```python
class EdgeDegreeEmbeddingNetwork(torch.nn.Module):
    def __init__(
        self, irreps_node_embedding, irreps_edge_attr, fc_neurons, avg_aggregate_num
    ): ...
```

### EdgeDegreeEmbeddingNetwork().forward

[Show source in equiformer_type_0_periodic.py:726](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L726)

#### Signature

```python
def forward(self, node_input, edge_attr, edge_scalars, edge_src, edge_dst, batch): ...
```



## EquiformerEnergyPeriodic

[Show source in equiformer_type_0_periodic.py:737](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L737)

#### Signature

```python
class EquiformerEnergyPeriodic(torch.nn.Module):
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

### EquiformerEnergyPeriodic().build_blocks

[Show source in equiformer_type_0_periodic.py:811](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L811)

#### Signature

```python
def build_blocks(self): ...
```

### EquiformerEnergyPeriodic().forward

[Show source in equiformer_type_0_periodic.py:865](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L865)

#### Signature

```python
def forward(self, pos, batch, node_atom, edge_index, **kwargs) -> torch.Tensor: ...
```

### EquiformerEnergyPeriodic().forward_with_gathered_index

[Show source in equiformer_type_0_periodic.py:893](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L893)

#### Signature

```python
def forward_with_gathered_index(
    self, pos, batch, node_atom, edge_index, periodic_index_mapping, **kwargs
) -> torch.Tensor: ...
```

### EquiformerEnergyPeriodic().no_weight_decay

[Show source in equiformer_type_0_periodic.py:844](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L844)

#### Signature

```python
@torch.jit.ignore
def no_weight_decay(self): ...
```



## FeedForwardNetwork

[Show source in equiformer_type_0_periodic.py:538](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L538)

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

[Show source in equiformer_type_0_periodic.py:567](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L567)

#### Signature

```python
def forward(self, node_input, node_attr, **kwargs): ...
```



## FullyConnectedTensorProductRescaleNorm

[Show source in equiformer_type_0_periodic.py:79](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L79)

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

[Show source in equiformer_type_0_periodic.py:93](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L93)

#### Signature

```python
def forward(self, x, y, batch, weight=None): ...
```



## FullyConnectedTensorProductRescaleNormSwishGate

[Show source in equiformer_type_0_periodic.py:99](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L99)

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

[Show source in equiformer_type_0_periodic.py:122](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L122)

#### Signature

```python
def forward(self, x, y, batch, weight=None): ...
```



## FullyConnectedTensorProductRescaleSwishGate

[Show source in equiformer_type_0_periodic.py:129](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L129)

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

[Show source in equiformer_type_0_periodic.py:152](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L152)

#### Signature

```python
def forward(self, x, y, weight=None): ...
```



## GraphAttention

[Show source in equiformer_type_0_periodic.py:404](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L404)

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

[Show source in equiformer_type_0_periodic.py:531](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L531)

#### Signature

```python
def extra_repr(self): ...
```

### GraphAttention().forward

[Show source in equiformer_type_0_periodic.py:483](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L483)

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

[Show source in equiformer_type_0_periodic.py:671](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L671)

#### Signature

```python
class NodeEmbeddingNetwork(torch.nn.Module):
    def __init__(self, irreps_node_embedding, max_atom_type, bias=True): ...
```

### NodeEmbeddingNetwork().forward

[Show source in equiformer_type_0_periodic.py:683](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L683)

`node_atom` is a LongTensor.

#### Signature

```python
def forward(self, node_atom): ...
```



## ScaledScatter

[Show source in equiformer_type_0_periodic.py:694](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L694)

#### Signature

```python
class ScaledScatter(torch.nn.Module):
    def __init__(self, avg_aggregate_num): ...
```

### ScaledScatter().extra_repr

[Show source in equiformer_type_0_periodic.py:706](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L706)

#### Signature

```python
def extra_repr(self): ...
```

### ScaledScatter().forward

[Show source in equiformer_type_0_periodic.py:700](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L700)

#### Signature

```python
def forward(self, x, index, **kwargs): ...
```



## SeparableFCTP

[Show source in equiformer_type_0_periodic.py:187](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L187)

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

[Show source in equiformer_type_0_periodic.py:235](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L235)

Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
self.dtp_rad(`edge_scalars`).

#### Signature

```python
def forward(self, node_input, edge_attr, edge_scalars, batch=None, **kwargs): ...
```



## SmoothLeakyReLU

[Show source in equiformer_type_0_periodic.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L55)

#### Signature

```python
class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.2): ...
```

### SmoothLeakyReLU().extra_repr

[Show source in equiformer_type_0_periodic.py:67](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L67)

#### Signature

```python
def extra_repr(self): ...
```

### SmoothLeakyReLU().forward

[Show source in equiformer_type_0_periodic.py:61](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L61)

#### Signature

```python
def forward(self, x): ...
```



## TransBlock

[Show source in equiformer_type_0_periodic.py:576](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L576)

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

[Show source in equiformer_type_0_periodic.py:640](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L640)

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

[Show source in equiformer_type_0_periodic.py:253](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L253)

Reshape vectors of shape [N, irreps_mid] to vectors of shape
[N, num_heads, irreps_head].

#### Signature

```python
class Vec2AttnHeads(torch.nn.Module):
    def __init__(self, irreps_head, num_heads): ...
```

### Vec2AttnHeads().forward

[Show source in equiformer_type_0_periodic.py:273](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L273)

#### Signature

```python
def forward(self, x): ...
```



## DepthwiseTensorProduct

[Show source in equiformer_type_0_periodic.py:158](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L158)

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



## get_mul_0

[Show source in equiformer_type_0_periodic.py:71](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L71)

#### Signature

```python
def get_mul_0(irreps): ...
```



## get_norm_layer

[Show source in equiformer_type_0_periodic.py:40](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_0_periodic.py#L40)

#### Signature

```python
def get_norm_layer(norm_type): ...
```