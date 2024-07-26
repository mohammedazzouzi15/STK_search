# Schnet

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Schnet

> Auto-generated documentation for [src.stk_search.geom3d.models.SchNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py) module.

- [Schnet](#schnet)
  - [CFConv](#cfconv)
    - [CFConv().forward](#cfconv()forward)
    - [CFConv().message](#cfconv()message)
    - [CFConv().reset_parameters](#cfconv()reset_parameters)
  - [GaussianSmearing](#gaussiansmearing)
    - [GaussianSmearing().forward](#gaussiansmearing()forward)
  - [InteractionBlock](#interactionblock)
    - [InteractionBlock().forward](#interactionblock()forward)
    - [InteractionBlock().reset_parameters](#interactionblock()reset_parameters)
  - [SchNet](#schnet)
    - [SchNet().forward](#schnet()forward)
    - [SchNet().forward_with_gathered_index](#schnet()forward_with_gathered_index)
    - [SchNet().reset_parameters](#schnet()reset_parameters)
  - [ShiftedSoftplus](#shiftedsoftplus)
    - [ShiftedSoftplus().forward](#shiftedsoftplus()forward)

## CFConv

[Show source in SchNet.py:206](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L206)

#### Signature

```python
class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff): ...
```

### CFConv().forward

[Show source in SchNet.py:221](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L221)

#### Signature

```python
def forward(self, x, edge_index, edge_weight, edge_attr): ...
```

### CFConv().message

[Show source in SchNet.py:231](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L231)

#### Signature

```python
def message(self, x_j, W): ...
```

### CFConv().reset_parameters

[Show source in SchNet.py:216](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L216)

#### Signature

```python
def reset_parameters(self): ...
```



## GaussianSmearing

[Show source in SchNet.py:235](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L235)

#### Signature

```python
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, gamma=None): ...
```

### GaussianSmearing().forward

[Show source in SchNet.py:245](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L245)

#### Signature

```python
def forward(self, dist): ...
```



## InteractionBlock

[Show source in SchNet.py:174](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L174)

#### Signature

```python
class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff): ...
```

### InteractionBlock().forward

[Show source in SchNet.py:199](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L199)

#### Signature

```python
def forward(self, x, edge_index, edge_weight, edge_attr): ...
```

### InteractionBlock().reset_parameters

[Show source in SchNet.py:190](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L190)

#### Signature

```python
def reset_parameters(self): ...
```



## SchNet

[Show source in SchNet.py:19](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L19)

#### Signature

```python
class SchNet(torch.nn.Module):
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        node_class=None,
        readout="mean",
        dipole=False,
        mean=None,
        std=None,
        atomref=None,
        gamma=None,
    ): ...
```

### SchNet().forward

[Show source in SchNet.py:89](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L89)

#### Signature

```python
def forward(self, z, pos, batch=None, edge_index=None, return_latent=False): ...
```

### SchNet().forward_with_gathered_index

[Show source in SchNet.py:136](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L136)

#### Signature

```python
def forward_with_gathered_index(
    self,
    gathered_z,
    pos,
    batch,
    edge_index,
    gathered_batch,
    periodic_index_mapping,
    return_latent=False,
): ...
```

### SchNet().reset_parameters

[Show source in SchNet.py:78](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L78)

#### Signature

```python
def reset_parameters(self): ...
```



## ShiftedSoftplus

[Show source in SchNet.py:250](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L250)

#### Signature

```python
class ShiftedSoftplus(torch.nn.Module):
    def __init__(self): ...
```

### ShiftedSoftplus().forward

[Show source in SchNet.py:255](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SchNet.py#L255)

#### Signature

```python
def forward(self, x): ...
```