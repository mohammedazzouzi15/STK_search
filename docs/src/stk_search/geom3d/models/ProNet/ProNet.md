# Pronet

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Pronet](./index.md#pronet) / Pronet

> Auto-generated documentation for [src.stk_search.geom3d.models.ProNet.ProNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py) module.

- [Pronet](#pronet)
  - [EdgeGraphConv](#edgegraphconv)
    - [EdgeGraphConv().forward](#edgegraphconv()forward)
    - [EdgeGraphConv().message](#edgegraphconv()message)
    - [EdgeGraphConv().message_and_aggregate](#edgegraphconv()message_and_aggregate)
    - [EdgeGraphConv().reset_parameters](#edgegraphconv()reset_parameters)
  - [InteractionBlock](#interactionblock)
    - [InteractionBlock().forward](#interactionblock()forward)
    - [InteractionBlock().reset_parameters](#interactionblock()reset_parameters)
  - [Linear](#linear)
    - [Linear().forward](#linear()forward)
    - [Linear().reset_parameters](#linear()reset_parameters)
  - [ProNet](#pronet)
    - [ProNet().forward](#pronet()forward)
    - [ProNet().num_params](#pronet()num_params)
    - [ProNet().pos_emb](#pronet()pos_emb)
    - [ProNet().reset_parameters](#pronet()reset_parameters)
  - [TwoLinear](#twolinear)
    - [TwoLinear().forward](#twolinear()forward)
    - [TwoLinear().reset_parameters](#twolinear()reset_parameters)
  - [swish](#swish)

## EdgeGraphConv

[Show source in ProNet.py:112](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L112)

Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

The difference is that this module performs Hadamard product between node feature and edge feature

Parameters
----------
in_channels (int)
out_channels (int)

#### Signature

```python
class EdgeGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels): ...
```

### EdgeGraphConv().forward

[Show source in ProNet.py:138](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L138)

#### Signature

```python
def forward(self, x, edge_index, edge_weight, size=None): ...
```

### EdgeGraphConv().message

[Show source in ProNet.py:144](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L144)

#### Signature

```python
def message(self, x_j, edge_weight): ...
```

### EdgeGraphConv().message_and_aggregate

[Show source in ProNet.py:147](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L147)

#### Signature

```python
def message_and_aggregate(self, adj_t, x): ...
```

### EdgeGraphConv().reset_parameters

[Show source in ProNet.py:134](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L134)

#### Signature

```python
def reset_parameters(self): ...
```



## InteractionBlock

[Show source in ProNet.py:151](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L151)

#### Signature

```python
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        output_channels,
        num_radial,
        num_spherical,
        num_layers,
        mid_emb,
        act=swish,
        num_pos_emb=16,
        dropout=0,
        level="allatom",
    ): ...
```

#### See also

- [swish](#swish)

### InteractionBlock().forward

[Show source in ProNet.py:223](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L223)

#### Signature

```python
def forward(self, x, feature0, feature1, pos_emb, edge_index, batch): ...
```

### InteractionBlock().reset_parameters

[Show source in ProNet.py:199](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L199)

#### Signature

```python
def reset_parameters(self): ...
```



## Linear

[Show source in ProNet.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L31)

A linear method encapsulation similar to PyG's

Parameters
----------
in_channels (int)
out_channels (int)
bias (int)
weight_initializer (string): (glorot or zeros)

#### Signature

```python
class Linear(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, bias=True, weight_initializer="glorot"
    ): ...
```

### Linear().forward

[Show source in ProNet.py:67](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L67)

#### Signature

```python
def forward(self, x): ...
```

### Linear().reset_parameters

[Show source in ProNet.py:59](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L59)

#### Signature

```python
def reset_parameters(self): ...
```



## ProNet

[Show source in ProNet.py:257](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L257)

The ProNet from the "Learning Protein Representations via Complete 3D Graph Networks" paper.

#### Arguments

- `level` - (str, optional): The level of protein representations. It could be :obj:`aminoacid`, obj:`backbone`, and :obj:`allatom`. (default: :obj:`aminoacid`)
- `num_blocks` *int, optional* - Number of building blocks. (default: :obj:`4`)
- `hidden_channels` *int, optional* - Hidden embedding size. (default: :obj:`128`)
- `out_channels` *int, optional* - Size of each output sample. (default: :obj:`1`)
- `mid_emb` *int, optional* - Embedding size used for geometric features. (default: :obj:`64`)
- `num_radial` *int, optional* - Number of radial basis functions. (default: :obj:`6`)
- `num_spherical` *int, optional* - Number of spherical harmonics. (default: :obj:`2`)
- `cutoff` *float, optional* - Cutoff distance for interatomic interactions. (default: :obj:`10.0`)
- `max_num_neighbors` *int, optional* - Max number of neighbors during graph construction. (default: :obj:`32`)
- `int_emb_layers` *int, optional* - Number of embedding layers in the interaction block. (default: :obj:`3`)
- `out_layers` *int, optional* - Number of layers for features after interaction blocks. (default: :obj:`2`)
- `num_pos_emb` *int, optional* - Number of positional embeddings. (default: :obj:`16`)
- `dropout` *float, optional* - Dropout. (default: :obj:`0`)
- `data_augment_eachlayer` *bool, optional* - Data augmentation tricks. If set to :obj:`True`, will add noise to the node features before each interaction block. (default: :obj:`False`)
- `euler_noise` *bool, optional* - Data augmentation tricks. If set to :obj:`True`, will add noise to Euler angles. (default: :obj:`False`)

#### Signature

```python
class ProNet(nn.Module):
    def __init__(
        self,
        level="aminoacid",
        num_blocks=4,
        hidden_channels=128,
        out_channels=1,
        mid_emb=64,
        num_radial=6,
        num_spherical=2,
        cutoff=10.0,
        max_num_neighbors=32,
        int_emb_layers=3,
        out_layers=2,
        num_pos_emb=16,
        dropout=0,
        data_augment_eachlayer=False,
        euler_noise=False,
    ): ...
```

### ProNet().forward

[Show source in ProNet.py:366](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L366)

#### Signature

```python
def forward(self, batch_data): ...
```

### ProNet().num_params

[Show source in ProNet.py:472](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L472)

#### Signature

```python
@property
def num_params(self): ...
```

### ProNet().pos_emb

[Show source in ProNet.py:354](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L354)

#### Signature

```python
def pos_emb(self, edge_index, num_pos_emb=16): ...
```

### ProNet().reset_parameters

[Show source in ProNet.py:346](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L346)

#### Signature

```python
def reset_parameters(self): ...
```



## TwoLinear

[Show source in ProNet.py:72](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L72)

A layer with two linear modules

Parameters
----------
in_channels (int)
middle_channels (int)
out_channels (int)
bias (bool)
act (bool)

#### Signature

```python
class TwoLinear(torch.nn.Module):
    def __init__(
        self, in_channels, middle_channels, out_channels, bias=False, act=False
    ): ...
```

### TwoLinear().forward

[Show source in ProNet.py:102](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L102)

#### Signature

```python
def forward(self, x): ...
```

### TwoLinear().reset_parameters

[Show source in ProNet.py:98](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L98)

#### Signature

```python
def reset_parameters(self): ...
```



## swish

[Show source in ProNet.py:27](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ProNet/ProNet.py#L27)

#### Signature

```python
def swish(x): ...
```