# Spherenet

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Spherenet

> Auto-generated documentation for [src.stk_search.geom3d.models.SphereNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py) module.

- [Spherenet](#spherenet)
  - [SphereNet](#spherenet)
    - [SphereNet().forward](#spherenet()forward)
    - [SphereNet().reset_parameters](#spherenet()reset_parameters)
    - [SphereNet().triplets](#spherenet()triplets)
  - [emb](#emb)
    - [emb().forward](#emb()forward)
    - [emb().reset_parameters](#emb()reset_parameters)
  - [init](#init)
    - [init().forward](#init()forward)
    - [init().reset_parameters](#init()reset_parameters)
  - [update_e](#update_e)
    - [update_e().forward](#update_e()forward)
    - [update_e().reset_parameters](#update_e()reset_parameters)
  - [update_u](#update_u)
    - [update_u().forward](#update_u()forward)
  - [update_v](#update_v)
    - [update_v().forward](#update_v()forward)
    - [update_v().reset_parameters](#update_v()reset_parameters)

## SphereNet

[Show source in SphereNet.py:223](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L223)

The spherical message passing neural network SphereNet from the `"Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013>`_ paper.

#### Arguments

- `energy_and_force` *bool, optional* - If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
- `cutoff` *float, optional* - Cutoff distance for interatomic interactions. (default: :obj:`5.0`)
- `num_layers` *int, optional* - Number of building blocks. (default: :obj:`4`)
- `hidden_channels` *int, optional* - Hidden embedding size. (default: :obj:`128`)
- `out_channels` *int, optional* - Size of each output sample. (default: :obj:`1`)
- `int_emb_size` *int, optional* - Embedding size used for interaction triplets. (default: :obj:`64`)
- `basis_emb_size_dist` *int, optional* - Embedding size used in the basis transformation of distance. (default: :obj:`8`)
- `basis_emb_size_angle` *int, optional* - Embedding size used in the basis transformation of angle. (default: :obj:`8`)
- `basis_emb_size_torsion` *int, optional* - Embedding size used in the basis transformation of torsion. (default: :obj:`8`)
- `out_emb_channels` *int, optional* - Embedding size used for atoms in the output block. (default: :obj:`256`)
- `num_spherical` *int, optional* - Number of spherical harmonics. (default: :obj:`7`)
- `num_radial` *int, optional* - Number of radial basis functions. (default: :obj:`6`)
- `envelop_exponent` *int, optional* - Shape of the smooth cutoff. (default: :obj:`5`)
- `num_before_skip` *int, optional* - Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`1`)
- `num_after_skip` *int, optional* - Number of residual layers in the interaction blocks before the skip connection. (default: :obj:`2`)
- `num_output_layers` *int, optional* - Number of linear layers for the output blocks. (default: :obj:`3`)
- `act` - (function, optional): The activation funtion. (default: :obj:`swish`)
- `output_init` - (str, optional): The initialization fot the output. It could be :obj:`GlorotOrthogonal` and :obj:`zeros`. (default: :obj:`GlorotOrthogonal`)

#### Signature

```python
class SphereNet(torch.nn.Module):
    def __init__(
        self,
        energy_and_force=False,
        cutoff=5.0,
        num_layers=4,
        hidden_channels=128,
        out_channels=1,
        int_emb_size=64,
        basis_emb_size_dist=8,
        basis_emb_size_angle=8,
        basis_emb_size_torsion=8,
        out_emb_channels=256,
        num_spherical=7,
        num_radial=6,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_output_layers=3,
        act="swish",
        output_init="GlorotOrthogonal",
    ): ...
```

### SphereNet().forward

[Show source in SphereNet.py:419](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L419)

#### Signature

```python
def forward(self, z, pos, batch): ...
```

### SphereNet().reset_parameters

[Show source in SphereNet.py:323](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L323)

#### Signature

```python
def reset_parameters(self): ...
```

### SphereNet().triplets

[Show source in SphereNet.py:332](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L332)

Compute the diatance, angle, and torsion from geometric information.

#### Arguments

- `pos` - Geometric information for every node in the graph.
- `edgee_index` - Edge index of the graph.
- `number_nodes` - Number of nodes in the graph.
- `use_torsion` - If set to :obj:`True`, will return distance, angle and torsion, otherwise only return distance and angle (also retrun some useful index). (default: :obj:`False`)

#### Signature

```python
def triplets(self, pos, edge_index, num_nodes, use_torsion=False): ...
```



## emb

[Show source in SphereNet.py:20](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L20)

#### Signature

```python
class emb(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff, envelope_exponent): ...
```

### emb().forward

[Show source in SphereNet.py:33](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L33)

#### Signature

```python
def forward(self, dist, angle, torsion, idx_kj): ...
```

### emb().reset_parameters

[Show source in SphereNet.py:30](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L30)

#### Signature

```python
def reset_parameters(self): ...
```



## init

[Show source in SphereNet.py:40](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L40)

#### Signature

```python
class init(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act="swish"): ...
```

### init().forward

[Show source in SphereNet.py:56](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L56)

#### Signature

```python
def forward(self, x, emb, i, j): ...
```

### init().reset_parameters

[Show source in SphereNet.py:50](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L50)

#### Signature

```python
def reset_parameters(self): ...
```



## update_e

[Show source in SphereNet.py:66](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L66)

#### Signature

```python
class update_e(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size_dist,
        basis_emb_size_angle,
        basis_emb_size_torsion,
        num_spherical,
        num_radial,
        num_before_skip,
        num_after_skip,
        act="swish",
    ): ...
```

### update_e().forward

[Show source in SphereNet.py:137](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L137)

#### Signature

```python
def forward(self, x, emb, idx_kj, idx_ji): ...
```

### update_e().reset_parameters

[Show source in SphereNet.py:112](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L112)

#### Signature

```python
def reset_parameters(self): ...
```



## update_u

[Show source in SphereNet.py:214](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L214)

#### Signature

```python
class update_u(torch.nn.Module):
    def __init__(self): ...
```

### update_u().forward

[Show source in SphereNet.py:218](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L218)

#### Signature

```python
def forward(self, u, v, batch): ...
```



## update_v

[Show source in SphereNet.py:172](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L172)

#### Signature

```python
class update_v(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        out_emb_channels,
        out_channels,
        num_output_layers,
        act,
        output_init,
    ): ...
```

### update_v().forward

[Show source in SphereNet.py:204](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L204)

#### Signature

```python
def forward(self, e, i): ...
```

### update_v().reset_parameters

[Show source in SphereNet.py:194](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SphereNet.py#L194)

#### Signature

```python
def reset_parameters(self): ...
```