# Cdconv

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Cdconv

> Auto-generated documentation for [src.stk_search.geom3d.models.CDConv](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py) module.

- [Cdconv](#cdconv)
  - [AvgPooling](#avgpooling)
    - [AvgPooling().forward](#avgpooling()forward)
  - [BasicBlock](#basicblock)
    - [BasicBlock().forward](#basicblock()forward)
  - [CDConv](#cdconv)
    - [CDConv().forward](#cdconv()forward)
    - [CDConv().message](#cdconv()message)
    - [CDConv().reset_parameters](#cdconv()reset_parameters)
  - [CDConv](#cdconv-1)
    - [CDConv().forward](#cdconv()forward-1)
  - [Linear](#linear)
    - [Linear().forward](#linear()forward)
  - [MLP](#mlp)
    - [MLP().forward](#mlp()forward)
  - [MaxPooling](#maxpooling)
    - [MaxPooling().forward](#maxpooling()forward)
  - [WeightNet](#weightnet)
    - [WeightNet().forward](#weightnet()forward)
    - [WeightNet().reset_parameters](#weightnet()reset_parameters)
  - [kaiming_uniform](#kaiming_uniform)

## AvgPooling

[Show source in CDConv.py:171](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L171)

#### Signature

```python
class AvgPooling(nn.Module):
    def __init__(self): ...
```

### AvgPooling().forward

[Show source in CDConv.py:175](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L175)

#### Signature

```python
def forward(self, x, pos, seq, ori, batch): ...
```



## BasicBlock

[Show source in CDConv.py:251](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L251)

#### Signature

```python
class BasicBlock(nn.Module):
    def __init__(
        self,
        r: float,
        l: float,
        kernel_channels,
        in_channels: int,
        out_channels: int,
        base_width: float = 16.0,
        batch_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        leakyrelu_negative_slope: float = 0.1,
        momentum: float = 0.2,
    ) -> nn.Module: ...
```

### BasicBlock().forward

[Show source in CDConv.py:296](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L296)

#### Signature

```python
def forward(self, x, pos, seq, ori, batch): ...
```



## CDConv

[Show source in CDConv.py:74](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L74)

#### Signature

```python
class CDConv(MessagePassing):
    def __init__(
        self,
        r: float,
        l: float,
        kernel_channels,
        in_channels: int,
        out_channels: int,
        add_self_loops: bool = True,
        **kwargs
    ): ...
```

### CDConv().forward

[Show source in CDConv.py:95](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L95)

#### Signature

```python
def forward(
    self, x: OptTensor, pos: Tensor, seq: Tensor, ori: Tensor, batch: Tensor
) -> Tensor: ...
```

### CDConv().message

[Show source in CDConv.py:112](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L112)

#### Signature

```python
def message(
    self,
    x_j: Optional[Tensor],
    pos_i: Tensor,
    pos_j: Tensor,
    seq_i: Tensor,
    seq_j: Tensor,
    ori_i: Tensor,
    ori_j: Tensor,
) -> Tensor: ...
```

### CDConv().reset_parameters

[Show source in CDConv.py:91](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L91)

#### Signature

```python
def reset_parameters(self): ...
```



## CDConv

[Show source in CDConv.py:303](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L303)

#### Signature

```python
class CDConv(nn.Module):
    def __init__(
        self,
        geometric_radii: List[float],
        sequential_kernel_size: float,
        kernel_channels: List[int],
        channels: List[int],
        base_width: float = 16.0,
        embedding_dim: int = 16,
        batch_norm: bool = True,
        dropout: float = 0.2,
        bias: bool = False,
        num_classes: int = 1195,
    ) -> nn.Module: ...
```

### CDConv().forward

[Show source in CDConv.py:354](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L354)

#### Signature

```python
def forward(self, data): ...
```



## Linear

[Show source in CDConv.py:192](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L192)

#### Signature

```python
class Linear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
        bias: bool = False,
        leakyrelu_negative_slope: float = 0.1,
        momentum: float = 0.2,
    ) -> nn.Module: ...
```

### Linear().forward

[Show source in CDConv.py:211](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L211)

#### Signature

```python
def forward(self, x): ...
```



## MLP

[Show source in CDConv.py:214](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L214)

#### Signature

```python
class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        batch_norm: bool,
        dropout: float = 0.0,
        bias: bool = True,
        leakyrelu_negative_slope: float = 0.2,
        momentum: float = 0.2,
    ) -> nn.Module: ...
```

### MLP().forward

[Show source in CDConv.py:248](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L248)

#### Signature

```python
def forward(self, input): ...
```



## MaxPooling

[Show source in CDConv.py:151](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L151)

#### Signature

```python
class MaxPooling(nn.Module):
    def __init__(self): ...
```

### MaxPooling().forward

[Show source in CDConv.py:155](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L155)

#### Signature

```python
def forward(self, x, pos, seq, ori, batch): ...
```



## WeightNet

[Show source in CDConv.py:35](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L35)

#### Signature

```python
class WeightNet(nn.Module):
    def __init__(self, l: int, kernel_channels): ...
```

### WeightNet().forward

[Show source in CDConv.py:63](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L63)

#### Signature

```python
def forward(self, input, idx): ...
```

### WeightNet().reset_parameters

[Show source in CDConv.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L55)

#### Signature

```python
def reset_parameters(self): ...
```



## kaiming_uniform

[Show source in CDConv.py:25](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CDConv.py#L25)

#### Signature

```python
def kaiming_uniform(tensor, size): ...
```