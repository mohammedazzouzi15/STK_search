# Molecule Gnn Model

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Molecule Gnn Model

> Auto-generated documentation for [src.stk_search.geom3d.models.molecule_gnn_model](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py) module.

- [Molecule Gnn Model](#molecule-gnn-model)
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
  - [GNN](#gnn)
    - [GNN().forward](#gnn()forward)
  - [GNN_graphpred](#gnn_graphpred)
    - [GNN_graphpred().forward](#gnn_graphpred()forward)
    - [GNN_graphpred().from_pretrained](#gnn_graphpred()from_pretrained)
    - [GNN_graphpred().get_graph_representation](#gnn_graphpred()get_graph_representation)
  - [GraphSAGEConv](#graphsageconv)
    - [GraphSAGEConv().forward](#graphsageconv()forward)
    - [GraphSAGEConv().message](#graphsageconv()message)
    - [GraphSAGEConv().update](#graphsageconv()update)

## GATConv

[Show source in molecule_gnn_model.py:65](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L65)

#### Signature

```python
class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"): ...
```

### GATConv().forward

[Show source in molecule_gnn_model.py:85](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L85)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GATConv().message

[Show source in molecule_gnn_model.py:91](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L91)

#### Signature

```python
def message(self, edge_index, x_i, x_j, edge_attr): ...
```

### GATConv().reset_parameters

[Show source in molecule_gnn_model.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L81)

#### Signature

```python
def reset_parameters(self): ...
```

### GATConv().update

[Show source in molecule_gnn_model.py:103](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L103)

#### Signature

```python
def update(self, aggr_out): ...
```



## GCNConv

[Show source in molecule_gnn_model.py:35](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L35)

#### Signature

```python
class GCNConv(MessagePassing):
    def __init__(self, emb_dim): ...
```

### GCNConv().forward

[Show source in molecule_gnn_model.py:43](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L43)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GCNConv().message

[Show source in molecule_gnn_model.py:58](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L58)

#### Signature

```python
def message(self, x_j, edge_attr, norm): ...
```

### GCNConv().update

[Show source in molecule_gnn_model.py:61](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L61)

#### Signature

```python
def update(self, aggr_out): ...
```



## GINConv

[Show source in molecule_gnn_model.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L13)

#### Signature

```python
class GINConv(MessagePassing):
    def __init__(self, emb_dim): ...
```

### GINConv().forward

[Show source in molecule_gnn_model.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L22)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GINConv().message

[Show source in molecule_gnn_model.py:28](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L28)

#### Signature

```python
def message(self, x_j, edge_attr): ...
```

### GINConv().update

[Show source in molecule_gnn_model.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L31)

#### Signature

```python
def update(self, aggr_out): ...
```



## GNN

[Show source in molecule_gnn_model.py:132](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L132)

#### Signature

```python
class GNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"): ...
```

### GNN().forward

[Show source in molecule_gnn_model.py:162](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L162)

#### Signature

```python
def forward(self, *argv): ...
```



## GNN_graphpred

[Show source in molecule_gnn_model.py:200](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L200)

#### Signature

```python
class GNN_graphpred(nn.Module):
    def __init__(self, args, num_tasks, molecule_model=None): ...
```

### GNN_graphpred().forward

[Show source in molecule_gnn_model.py:262](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L262)

#### Signature

```python
def forward(self, *argv): ...
```

### GNN_graphpred().from_pretrained

[Show source in molecule_gnn_model.py:237](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L237)

#### Signature

```python
def from_pretrained(self, model_file): ...
```

### GNN_graphpred().get_graph_representation

[Show source in molecule_gnn_model.py:241](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L241)

#### Signature

```python
def get_graph_representation(self, *argv): ...
```



## GraphSAGEConv

[Show source in molecule_gnn_model.py:109](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L109)

#### Signature

```python
class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"): ...
```

### GraphSAGEConv().forward

[Show source in molecule_gnn_model.py:119](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L119)

#### Signature

```python
def forward(self, x, edge_index, edge_attr): ...
```

### GraphSAGEConv().message

[Show source in molecule_gnn_model.py:125](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L125)

#### Signature

```python
def message(self, x_j, edge_attr): ...
```

### GraphSAGEConv().update

[Show source in molecule_gnn_model.py:128](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/molecule_gnn_model.py#L128)

#### Signature

```python
def update(self, aggr_out): ...
```