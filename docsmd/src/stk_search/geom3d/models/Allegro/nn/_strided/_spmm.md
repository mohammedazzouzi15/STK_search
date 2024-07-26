# Spmm

[stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Allegro](../../index.md#allegro) / [Nn](../index.md#nn) / [Strided](./index.md#strided) / Spmm

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._strided._spmm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py) module.

- [Spmm](#spmm)
  - [ExplicitGradSpmm](#explicitgradspmm)
    - [ExplicitGradSpmm().forward](#explicitgradspmm()forward)
  - [ExplicitGradSpmmCOO](#explicitgradspmmcoo)
    - [ExplicitGradSpmmCOO().forward](#explicitgradspmmcoo()forward)
  - [ExplicitGradSpmmCSR](#explicitgradspmmcsr)
    - [ExplicitGradSpmmCSR().forward](#explicitgradspmmcsr()forward)
  - [ExplicitGradSpmm](#explicitgradspmm-1)

## ExplicitGradSpmm

[Show source in _spmm.py:148](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L148)

#### Signature

```python
class ExplicitGradSpmm(torch.nn.Module):
    def __init__(self, mat): ...
```

### ExplicitGradSpmm().forward

[Show source in _spmm.py:153](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L153)

#### Signature

```python
def forward(self, x): ...
```



## ExplicitGradSpmmCOO

[Show source in _spmm.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L38)

#### Signature

```python
class ExplicitGradSpmmCOO(torch.nn.Module):
    def __init__(self, mat: torch.Tensor): ...
```

### ExplicitGradSpmmCOO().forward

[Show source in _spmm.py:52](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L52)

#### Signature

```python
def forward(self, x): ...
```



## ExplicitGradSpmmCSR

[Show source in _spmm.py:94](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L94)

#### Signature

```python
class ExplicitGradSpmmCSR(torch.nn.Module):
    def __init__(self, mat: torch.Tensor): ...
```

### ExplicitGradSpmmCSR().forward

[Show source in _spmm.py:108](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L108)

#### Signature

```python
def forward(self, x): ...
```



## ExplicitGradSpmm

[Show source in _spmm.py:135](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_spmm.py#L135)

#### Signature

```python
def ExplicitGradSpmm(mat): ...
```