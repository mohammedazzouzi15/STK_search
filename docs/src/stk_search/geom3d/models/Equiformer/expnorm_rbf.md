# Expnorm Rbf

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Expnorm Rbf

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.expnorm_rbf](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/expnorm_rbf.py) module.

- [Expnorm Rbf](#expnorm-rbf)
  - [CosineCutoff](#cosinecutoff)
    - [CosineCutoff().forward](#cosinecutoff()forward)
  - [ExpNormalSmearing](#expnormalsmearing)
    - [ExpNormalSmearing().forward](#expnormalsmearing()forward)
    - [ExpNormalSmearing().reset_parameters](#expnormalsmearing()reset_parameters)

## CosineCutoff

[Show source in expnorm_rbf.py:5](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/expnorm_rbf.py#L5)

#### Signature

```python
class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0): ...
```

### CosineCutoff().forward

[Show source in expnorm_rbf.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/expnorm_rbf.py#L11)

#### Signature

```python
def forward(self, distances): ...
```



## ExpNormalSmearing

[Show source in expnorm_rbf.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/expnorm_rbf.py#L37)

#### Signature

```python
class ExpNormalSmearing(torch.nn.Module):
    def __init__(
        self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False
    ): ...
```

### ExpNormalSmearing().forward

[Show source in expnorm_rbf.py:73](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/expnorm_rbf.py#L73)

#### Signature

```python
def forward(self, dist): ...
```

### ExpNormalSmearing().reset_parameters

[Show source in expnorm_rbf.py:68](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/expnorm_rbf.py#L68)

#### Signature

```python
def reset_parameters(self): ...
```