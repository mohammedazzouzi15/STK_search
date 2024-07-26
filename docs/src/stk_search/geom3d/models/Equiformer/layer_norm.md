# Layer Norm

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Layer Norm

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.layer_norm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py) module.

#### Attributes

- `rot` - Check equivariant: -o3.rand_matrix()


- [Layer Norm](#layer-norm)
  - [EquivariantLayerNorm](#equivariantlayernorm)
    - [EquivariantLayerNorm().forward](#equivariantlayernorm()forward)
  - [EquivariantLayerNormV2](#equivariantlayernormv2)
    - [EquivariantLayerNormV2().forward](#equivariantlayernormv2()forward)
  - [EquivariantLayerNormV3](#equivariantlayernormv3)
    - [EquivariantLayerNormV3().forward](#equivariantlayernormv3()forward)
  - [EquivariantLayerNormV4](#equivariantlayernormv4)
    - [EquivariantLayerNormV4().forward](#equivariantlayernormv4()forward)

## EquivariantLayerNorm

[Show source in layer_norm.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L12)

#### Signature

```python
class EquivariantLayerNorm(torch.nn.Module):
    def __init__(self, irreps_in, eps=1e-05): ...
```

### EquivariantLayerNorm().forward

[Show source in layer_norm.py:29](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L29)

Assume `f_in` is of shape [N, C].

#### Signature

```python
def forward(self, f_in, **kwargs): ...
```



## EquivariantLayerNormV2

[Show source in layer_norm.py:62](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L62)

#### Signature

```python
class EquivariantLayerNormV2(nn.Module):
    def __init__(self, irreps, eps=1e-05, affine=True, normalization="component"): ...
```

### EquivariantLayerNormV2().forward

[Show source in layer_norm.py:89](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L89)

#### Signature

```python
@torch.cuda.amp.autocast(enabled=False)
def forward(self, node_input, **kwargs): ...
```



## EquivariantLayerNormV3

[Show source in layer_norm.py:155](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L155)

V2 + Centering for vectors of all degrees

#### Signature

```python
class EquivariantLayerNormV3(nn.Module):
    def __init__(self, irreps, eps=1e-05, affine=True, normalization="component"): ...
```

### EquivariantLayerNormV3().forward

[Show source in layer_norm.py:185](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L185)

#### Signature

```python
def forward(self, node_input, **kwargs): ...
```



## EquivariantLayerNormV4

[Show source in layer_norm.py:235](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L235)

V3 + Learnable mean shift

#### Signature

```python
class EquivariantLayerNormV4(nn.Module):
    def __init__(self, irreps, eps=1e-05, affine=True, normalization="component"): ...
```

### EquivariantLayerNormV4().forward

[Show source in layer_norm.py:274](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/layer_norm.py#L274)

#### Signature

```python
def forward(self, node_input, **kwargs): ...
```