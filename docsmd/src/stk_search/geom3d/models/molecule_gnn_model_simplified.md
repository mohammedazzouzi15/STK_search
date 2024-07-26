# Molecule Gnn Model Simplified

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Molecule Gnn Model Simplified

> Auto-generated documentation for [src.stk_search.geom3d.models.molecule_gnn_model_simplified](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py) module.

- [Molecule Gnn Model Simplified](#molecule-gnn-model-simplified)
  - [GATConv](#gatconv)
    - [GATConv().forward](#gatconv()forward)
    - [GATConv().message](#gatconv()message)
    - [GATConv().reset_parameters](#gatconv()reset_parameters)
    - [GATConv().update](#gatconv()update)
  - [GCNConv](#gcnconv)
    - [GCNConv().forward](#gcnconv()forward)
    - [GCNConv().message](#gcnconv()message)
    - [GCNConv().update](#gcnconv()update)
  - [GINConv](#ginconv)
    - [GINConv().forward](#ginconv()forward)
    - [GINConv().message](#ginconv()message)
    - [GINConv().update](#ginconv()update)
  - [GNNSimplified](#gnnsimplified)
    - [GNNSimplified().forward](#gnnsimplified()forward)
  - [GraphSAGEConv](#graphsageconv)
    - [GraphSAGEConv().forward](#graphsageconv()forward)
    - [GraphSAGEConv().message](#graphsageconv()message)
    - [GraphSAGEConv().update](#graphsageconv()update)

## GATConv

[Show source in molecule_gnn_model_simplified.py:65](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L65)

#### Signature

```python
class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"): ...
```

### GATConv().forward

[Show source in molecule_gnn_model_simplified.py:85](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L85)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GATConv().message

[Show source in molecule_gnn_model_simplified.py:91](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L91)

#### Signature

```python
def message(self, edge_index, x_i, x_j, edge_attr): ...
```

### GATConv().reset_parameters

[Show source in molecule_gnn_model_simplified.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L81)

#### Signature

```python
def reset_parameters(self): ...
```

### GATConv().update

[Show source in molecule_gnn_model_simplified.py:103](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L103)

#### Signature

```python
def update(self, aggr_out): ...
```



## GCNConv

[Show source in molecule_gnn_model_simplified.py:35](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L35)

#### Signature

```python
class GCNConv(MessagePassing):
    def __init__(self, emb_dim): ...
```

### GCNConv().forward

[Show source in molecule_gnn_model_simplified.py:43](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L43)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GCNConv().message

[Show source in molecule_gnn_model_simplified.py:58](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L58)

#### Signature

```python
def message(self, x_j, edge_attr, norm): ...
```

### GCNConv().update

[Show source in molecule_gnn_model_simplified.py:61](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L61)

#### Signature

```python
def update(self, aggr_out): ...
```



## GINConv

[Show source in molecule_gnn_model_simplified.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L13)

#### Signature

```python
class GINConv(MessagePassing):
    def __init__(self, emb_dim): ...
```

### GINConv().forward

[Show source in molecule_gnn_model_simplified.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L22)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GINConv().message

[Show source in molecule_gnn_model_simplified.py:28](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L28)

#### Signature

```python
def message(self, x_j, edge_attr): ...
```

### GINConv().update

[Show source in molecule_gnn_model_simplified.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L31)

#### Signature

```python
def update(self, aggr_out): ...
```



## GNNSimplified

[Show source in molecule_gnn_model_simplified.py:132](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L132)

#### Signature

```python
class GNNSimplified(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"): ...
```

### GNNSimplified().forward

[Show source in molecule_gnn_model_simplified.py:162](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L162)

#### Signature

```python
def forward(self, *argv): ...
```



## GraphSAGEConv

[Show source in molecule_gnn_model_simplified.py:109](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L109)

#### Signature

```python
class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"): ...
```

### GraphSAGEConv().forward

[Show source in molecule_gnn_model_simplified.py:119](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L119)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GraphSAGEConv().message

[Show source in molecule_gnn_model_simplified.py:125](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L125)

#### Signature

```python
def message(self, x_j, edge_attr): ...
```

### GraphSAGEConv().update

[Show source in molecule_gnn_model_simplified.py:128](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model_simplified.py#L128)

#### Signature

```python
def update(self, aggr_out): ...
```