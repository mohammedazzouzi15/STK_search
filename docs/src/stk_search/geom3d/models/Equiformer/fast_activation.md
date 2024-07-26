# Fast Activation

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Fast Activation

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.fast_activation](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py) module.

- [Fast Activation](#fast-activation)
  - [Activation](#activation)
    - [Activation().extra_repr](#activation()extra_repr)
    - [Activation().forward](#activation()forward)
  - [Gate](#gate)
    - [Gate().forward](#gate()forward)
    - [Gate().irreps_in](#gate()irreps_in)
    - [Gate().irreps_out](#gate()irreps_out)

## Activation

[Show source in fast_activation.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L15)

Directly apply activation when irreps is type-0.

#### Signature

```python
class Activation(torch.nn.Module):
    def __init__(self, irreps_in, acts): ...
```

### Activation().extra_repr

[Show source in fast_activation.py:62](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L62)

#### Signature

```python
def extra_repr(self): ...
```

### Activation().forward

[Show source in fast_activation.py:68](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L68)

#### Signature

```python
def forward(self, features, dim=-1): ...
```



## Gate

[Show source in fast_activation.py:91](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L91)

1. Use `narrow` to split tensor.
2. Use [Activation](#activation) in this file.

#### Signature

```python
class Gate(torch.nn.Module):
    def __init__(
        self, irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated
    ): ...
```

### Gate().forward

[Show source in fast_activation.py:132](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L132)

#### Signature

```python
def forward(self, features): ...
```

### Gate().irreps_in

[Show source in fast_activation.py:151](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L151)

Input representations.

#### Signature

```python
@property
def irreps_in(self): ...
```

### Gate().irreps_out

[Show source in fast_activation.py:157](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/fast_activation.py#L157)

Output representations.

#### Signature

```python
@property
def irreps_out(self): ...
```