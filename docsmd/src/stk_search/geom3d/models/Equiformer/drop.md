# Drop

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Drop

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.drop](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py) module.

- [Drop](#drop)
  - [DropPath](#droppath)
    - [DropPath().extra_repr](#droppath()extra_repr)
    - [DropPath().forward](#droppath()forward)
  - [EquivariantDropout](#equivariantdropout)
    - [EquivariantDropout().forward](#equivariantdropout()forward)
  - [EquivariantScalarsDropout](#equivariantscalarsdropout)
    - [EquivariantScalarsDropout().extra_repr](#equivariantscalarsdropout()extra_repr)
    - [EquivariantScalarsDropout().forward](#equivariantscalarsdropout()forward)
  - [GraphDropPath](#graphdroppath)
    - [GraphDropPath().extra_repr](#graphdroppath()extra_repr)
    - [GraphDropPath().forward](#graphdroppath()forward)
  - [drop_path](#drop_path)

## DropPath

[Show source in drop.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L31)

Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

#### Signature

```python
class DropPath(nn.Module):
    def __init__(self, drop_prob=None): ...
```

### DropPath().extra_repr

[Show source in drop.py:41](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L41)

#### Signature

```python
def extra_repr(self): ...
```

### DropPath().forward

[Show source in drop.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L38)

#### Signature

```python
def forward(self, x): ...
```



## EquivariantDropout

[Show source in drop.py:68](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L68)

#### Signature

```python
class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob): ...
```

### EquivariantDropout().forward

[Show source in drop.py:79](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L79)

#### Signature

```python
def forward(self, x): ...
```



## EquivariantScalarsDropout

[Show source in drop.py:89](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L89)

#### Signature

```python
class EquivariantScalarsDropout(nn.Module):
    def __init__(self, irreps, drop_prob): ...
```

### EquivariantScalarsDropout().extra_repr

[Show source in drop.py:111](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L111)

#### Signature

```python
def extra_repr(self): ...
```

### EquivariantScalarsDropout().forward

[Show source in drop.py:96](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L96)

#### Signature

```python
def forward(self, x): ...
```



## GraphDropPath

[Show source in drop.py:45](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L45)

Consider batch for graph data when dropping paths.

#### Signature

```python
class GraphDropPath(nn.Module):
    def __init__(self, drop_prob=None): ...
```

### GraphDropPath().extra_repr

[Show source in drop.py:63](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L63)

#### Signature

```python
def extra_repr(self): ...
```

### GraphDropPath().forward

[Show source in drop.py:54](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L54)

#### Signature

```python
def forward(self, x, batch): ...
```



## drop_path

[Show source in drop.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/drop.py#L13)

Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
'survival rate' as the argument.

#### Signature

```python
def drop_path(x, drop_prob: float = 0.0, training: bool = False): ...
```