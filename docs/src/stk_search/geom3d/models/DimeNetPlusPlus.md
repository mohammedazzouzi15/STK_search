# Dimenetplusplus

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Dimenetplusplus

> Auto-generated documentation for [src.stk_search.geom3d.models.DimeNetPlusPlus](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py) module.

- [Dimenetplusplus](#dimenetplusplus)
  - [DimeNetPlusPlus](#dimenetplusplus)
    - [DimeNetPlusPlus().forward](#dimenetplusplus()forward)
    - [DimeNetPlusPlus().forward_with_gathered_index](#dimenetplusplus()forward_with_gathered_index)
    - [DimeNetPlusPlus().reset_parameters](#dimenetplusplus()reset_parameters)
    - [DimeNetPlusPlus().triplets](#dimenetplusplus()triplets)
  - [InteractionPPBlock](#interactionppblock)
    - [InteractionPPBlock().forward](#interactionppblock()forward)
    - [InteractionPPBlock().reset_parameters](#interactionppblock()reset_parameters)
  - [OutputPPBlock](#outputppblock)
    - [OutputPPBlock().forward](#outputppblock()forward)
    - [OutputPPBlock().reset_parameters](#outputppblock()reset_parameters)

## DimeNetPlusPlus

[Show source in DimeNetPlusPlus.py:164](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L164)

DimeNet++ implementation based on https://github.com/klicperajo/DimeNet.

#### Arguments

- `hidden_channels` *int* - Hidden embedding size.
- `out_channels` *int* - Size of each output sample.
- `num_blocks` *int* - Number of building blocks.
- `int_emb_size` *int* - Embedding size used for interaction triplets
- `basis_emb_size` *int* - Embedding size used in the basis transformation
- `out_emb_channels(int)` - Embedding size used for atoms in the output block
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
    - `(default` - :obj:`swish`)

#### Signature

```python
class DimeNetPlusPlus(torch.nn.Module):
    def __init__(
        self,
        node_class,
        hidden_channels=128,
        out_channels=1,
        num_blocks=4,
        int_emb_size=64,
        basis_emb_size=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act=swish,
        readout="add",
    ): ...
```

### DimeNetPlusPlus().forward

[Show source in DimeNetPlusPlus.py:285](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L285)

#### Signature

```python
def forward(
    self,
    z,
    pos,
    batch,
    edge_index=None,
    extract_representation=False,
    return_latent=False,
): ...
```

### DimeNetPlusPlus().forward_with_gathered_index

[Show source in DimeNetPlusPlus.py:326](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L326)

#### Signature

```python
def forward_with_gathered_index(
    self,
    gathered_z,
    pos,
    batch,
    edge_index,
    periodic_index_mapping,
    gathered_batch,
    extract_representation=False,
    return_latent=False,
): ...
```

### DimeNetPlusPlus().reset_parameters

[Show source in DimeNetPlusPlus.py:254](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L254)

#### Signature

```python
def reset_parameters(self): ...
```

### DimeNetPlusPlus().triplets

[Show source in DimeNetPlusPlus.py:262](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L262)

#### Signature

```python
def triplets(self, edge_index, num_nodes): ...
```



## InteractionPPBlock

[Show source in DimeNetPlusPlus.py:27](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L27)

#### Signature

```python
class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act=swish,
    ): ...
```

### InteractionPPBlock().forward

[Show source in DimeNetPlusPlus.py:90](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L90)

#### Signature

```python
def forward(self, x, rbf, sbf, idx_kj, idx_ji): ...
```

### InteractionPPBlock().reset_parameters

[Show source in DimeNetPlusPlus.py:69](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L69)

#### Signature

```python
def reset_parameters(self): ...
```



## OutputPPBlock

[Show source in DimeNetPlusPlus.py:122](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L122)

#### Signature

```python
class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_layers,
        act=swish,
    ): ...
```

### OutputPPBlock().forward

[Show source in DimeNetPlusPlus.py:152](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L152)

#### Signature

```python
def forward(self, x, rbf, i, num_nodes=None, extract_representation=False): ...
```

### OutputPPBlock().reset_parameters

[Show source in DimeNetPlusPlus.py:144](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DimeNetPlusPlus.py#L144)

#### Signature

```python
def reset_parameters(self): ...
```