# Dimenet

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Dimenet

> Auto-generated documentation for [src.stk_search.geom3d.models.DimeNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py) module.

- [Dimenet](#dimenet)
  - [BesselBasisLayer](#besselbasislayer)
    - [BesselBasisLayer().forward](#besselbasislayer()forward)
    - [BesselBasisLayer().reset_parameters](#besselbasislayer()reset_parameters)
  - [DimeNet](#dimenet)
    - [DimeNet().forward](#dimenet()forward)
    - [DimeNet().reset_parameters](#dimenet()reset_parameters)
    - [DimeNet().triplets](#dimenet()triplets)
  - [EmbeddingBlock](#embeddingblock)
    - [EmbeddingBlock().forward](#embeddingblock()forward)
    - [EmbeddingBlock().reset_parameters](#embeddingblock()reset_parameters)
  - [Envelope](#envelope)
    - [Envelope().forward](#envelope()forward)
  - [InteractionBlock](#interactionblock)
    - [InteractionBlock().forward](#interactionblock()forward)
    - [InteractionBlock().reset_parameters](#interactionblock()reset_parameters)
  - [OutputBlock](#outputblock)
    - [OutputBlock().forward](#outputblock()forward)
    - [OutputBlock().reset_parameters](#outputblock()reset_parameters)
  - [ResidualLayer](#residuallayer)
    - [ResidualLayer().forward](#residuallayer()forward)
    - [ResidualLayer().reset_parameters](#residuallayer()reset_parameters)
  - [SphericalBasisLayer](#sphericalbasislayer)
    - [SphericalBasisLayer().forward](#sphericalbasislayer()forward)
  - [glorot_orthogonal](#glorot_orthogonal)
  - [swish](#swish)

## BesselBasisLayer

[Show source in DimeNet.py:46](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L46)

#### Signature

```python
class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5): ...
```

### BesselBasisLayer().forward

[Show source in DimeNet.py:60](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L60)

#### Signature

```python
def forward(self, dist): ...
```

### BesselBasisLayer().reset_parameters

[Show source in DimeNet.py:56](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L56)

#### Signature

```python
def reset_parameters(self): ...
```



## DimeNet

[Show source in DimeNet.py:244](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L244)

The directional message passing neural network (DimeNet) from the
`"Directional Message Passing for Molecular Graphs"
<https://arxiv.org/abs/2003.03123>`_ paper.
DimeNet transforms messages based on the angle between them in a
rotation-equivariant fashion.
.. note

```python
For an example of using a pretrained DimeNet variant, see
`examples/qm9_pretrained_DimeNet.py
<https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
qm9_pretrained_DimeNet.py>`_.
```

#### Arguments

- `hidden_channels` *int* - Hidden embedding size.
- `out_channels` *int* - Size of each output sample.
- `num_blocks` *int* - Number of building blocks.
- `num_bilinear` *int* - Size of the bilinear layer tensor.
- `num_spherical` *int* - Number of spherical harmonics.
- `num_radial` *int* - Number of radial basis functions.
- `cutoff` - (float, optional): Cutoff distance for interatomic
    - `interactions.` *(default* - :obj:`5.0`)
- `envelope_exponent` *int, optional* - Shape of the smooth cutoff.
    - `(default` - :obj:`5`)
- `num_before_skip` - (int, optional): Number of residual layers in the
    interaction blocks before the skip connection. (default: :obj:`1`)
- `num_after_skip` - (int, optional): Number of residual layers in the
    interaction blocks after the skip connection. (default: :obj:`2`)
- `num_output_layers` - (int, optional): Number of linear layers for the
    output blocks. (default: :obj:`3`)
- `act` - (function, optional): The activation funtion.
    - `(default` - :obj:[swish](#swish))

#### Signature

```python
class DimeNet(nn.Module):
    def __init__(
        self,
        node_class,
        hidden_channels,
        out_channels,
        num_blocks,
        num_bilinear,
        num_spherical,
        num_radial,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
    ): ...
```

#### See also

- [swish](#swish)

### DimeNet().forward

[Show source in DimeNet.py:363](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L363)

#### Signature

```python
def forward(self, z, pos, batch=None): ...
```

### DimeNet().reset_parameters

[Show source in DimeNet.py:331](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L331)

#### Signature

```python
def reset_parameters(self): ...
```

### DimeNet().triplets

[Show source in DimeNet.py:340](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L340)

#### Signature

```python
def triplets(self, edge_index, num_nodes): ...
```



## EmbeddingBlock

[Show source in DimeNet.py:104](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L104)

#### Signature

```python
class EmbeddingBlock(torch.nn.Module):
    def __init__(self, node_class, num_radial, hidden_channels, act=swish): ...
```

#### See also

- [swish](#swish)

### EmbeddingBlock().forward

[Show source in DimeNet.py:120](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L120)

#### Signature

```python
def forward(self, x, rbf, i, j): ...
```

### EmbeddingBlock().reset_parameters

[Show source in DimeNet.py:115](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L115)

#### Signature

```python
def reset_parameters(self): ...
```



## Envelope

[Show source in DimeNet.py:30](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L30)

#### Signature

```python
class Envelope(torch.nn.Module):
    def __init__(self, exponent): ...
```

### Envelope().forward

[Show source in DimeNet.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L38)

#### Signature

```python
def forward(self, x): ...
```



## InteractionBlock

[Show source in DimeNet.py:145](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L145)

#### Signature

```python
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_bilinear,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ): ...
```

#### See also

- [swish](#swish)

### InteractionBlock().forward

[Show source in DimeNet.py:195](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L195)

#### Signature

```python
def forward(self, x, rbf, sbf, idx_kj, idx_ji): ...
```

### InteractionBlock().reset_parameters

[Show source in DimeNet.py:180](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L180)

#### Signature

```python
def reset_parameters(self): ...
```



## OutputBlock

[Show source in DimeNet.py:214](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L214)

#### Signature

```python
class OutputBlock(torch.nn.Module):
    def __init__(
        self, num_radial, hidden_channels, out_channels, num_layers, act=swish
    ): ...
```

#### See also

- [swish](#swish)

### OutputBlock().forward

[Show source in DimeNet.py:236](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L236)

#### Signature

```python
def forward(self, x, rbf, i, num_nodes=None): ...
```

### OutputBlock().reset_parameters

[Show source in DimeNet.py:229](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L229)

#### Signature

```python
def reset_parameters(self): ...
```



## ResidualLayer

[Show source in DimeNet.py:126](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L126)

#### Signature

```python
class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish): ...
```

#### See also

- [swish](#swish)

### ResidualLayer().forward

[Show source in DimeNet.py:141](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L141)

#### Signature

```python
def forward(self, x): ...
```

### ResidualLayer().reset_parameters

[Show source in DimeNet.py:135](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L135)

#### Signature

```python
def reset_parameters(self): ...
```



## SphericalBasisLayer

[Show source in DimeNet.py:65](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L65)

#### Signature

```python
class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0, envelope_exponent=5): ...
```

### SphericalBasisLayer().forward

[Show source in DimeNet.py:92](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L92)

#### Signature

```python
def forward(self, dist, angle, idx_kj): ...
```



## glorot_orthogonal

[Show source in DimeNet.py:23](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L23)

#### Signature

```python
def glorot_orthogonal(tensor, scale): ...
```



## swish

[Show source in DimeNet.py:19](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNet.py#L19)

#### Signature

```python
def swish(x): ...
```