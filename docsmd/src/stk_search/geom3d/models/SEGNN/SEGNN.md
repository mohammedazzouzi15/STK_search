# Segnn

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Segnn](./index.md#segnn) / Segnn

> Auto-generated documentation for [src.stk_search.geom3d.models.SEGNN.SEGNN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py) module.

- [Segnn](#segnn)
  - [SEGNN](#segnn)
    - [SEGNN().forward](#segnn()forward)
    - [SEGNN().message](#segnn()message)
    - [SEGNN().update](#segnn()update)
  - [SEGNNModel](#segnnmodel)
    - [SEGNNModel().forward](#segnnmodel()forward)
    - [SEGNNModel().forward_with_gathered_index](#segnnmodel()forward_with_gathered_index)

## SEGNN

[Show source in SEGNN.py:174](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L174)

E(3) equivariant message passing layer.

#### Signature

```python
class SEGNN(MessagePassing):
    def __init__(
        self,
        node_in_irreps,
        node_hidden_irreps,
        node_out_irreps,
        attr_irreps,
        norm,
        edge_inference,
    ): ...
```

### SEGNN().forward

[Show source in SEGNN.py:215](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L215)

Propagate messages along edges

#### Signature

```python
def forward(self, x, pos, edge_index, edge_dist, edge_attr, node_attr, batch): ...
```

### SEGNN().message

[Show source in SEGNN.py:229](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L229)

Create messages

#### Signature

```python
def message(self, x_i, x_j, edge_dist, edge_attr): ...
```

### SEGNN().update

[Show source in SEGNN.py:241](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L241)

Update note features

#### Signature

```python
def update(self, message, x, pos, node_attr): ...
```



## SEGNNModel

[Show source in SEGNN.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L17)

#### Signature

```python
class SEGNNModel(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        hidden_features,
        N,
        norm,
        lmax_h,
        lmax_pos=None,
        pool="avg",
        edge_inference=False,
    ): ...
```

### SEGNNModel().forward

[Show source in SEGNN.py:72](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L72)

#### Signature

```python
def forward(self, *argv): ...
```

### SEGNNModel().forward_with_gathered_index

[Show source in SEGNN.py:124](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/SEGNN.py#L124)

#### Signature

```python
def forward_with_gathered_index(
    self, x, pos, edge_index, batch, periodic_index_mapping, graph
): ...
```