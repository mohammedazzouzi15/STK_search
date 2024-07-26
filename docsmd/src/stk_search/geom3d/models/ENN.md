# Enn

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Enn

> Auto-generated documentation for [src.stk_search.geom3d.models.ENN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py) module.

- [Enn](#enn)
  - [ENN_S2S](#enn_s2s)
    - [ENN_S2S().forward](#enn_s2s()forward)
  - [GraphSoftmax](#graphsoftmax)
    - [GraphSoftmax().forward](#graphsoftmax()forward)
  - [Set2Set](#set2set)
    - [Set2Set().forward](#set2set()forward)

## ENN_S2S

[Show source in ENN.py:57](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py#L57)

#### Signature

```python
class ENN_S2S(nn.Module):
    def __init__(
        self,
        hidden_dim,
        gru_layer_num,
        enn_layer_num,
        set2set_processing_steps,
        set2set_num_layers,
        output_dim,
    ): ...
```

### ENN_S2S().forward

[Show source in ENN.py:78](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py#L78)

#### Signature

```python
def forward(self, data): ...
```



## GraphSoftmax

[Show source in ENN.py:14](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py#L14)

#### Signature

```python
class GraphSoftmax(nn.Module): ...
```

### GraphSoftmax().forward

[Show source in ENN.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py#L17)

#### Signature

```python
def forward(self, batch, input): ...
```



## Set2Set

[Show source in ENN.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py#L26)

#### Signature

```python
class Set2Set(nn.Module):
    def __init__(self, input_dim, processing_steps, num_layers): ...
```

### Set2Set().forward

[Show source in ENN.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/ENN.py#L37)

#### Signature

```python
def forward(self, x, batch): ...
```