# Egnn

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Egnn

> Auto-generated documentation for [src.stk_search.geom3d.models.EGNN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py) module.

- [Egnn](#egnn)
  - [EGNN](#egnn)
    - [EGNN().forward](#egnn()forward)
    - [EGNN().forward_with_gathered_index](#egnn()forward_with_gathered_index)
  - [E_GCL](#e_gcl)
    - [E_GCL().edge_model](#e_gcl()edge_model)
    - [E_GCL().forward](#e_gcl()forward)
    - [E_GCL().forward_with_gathered_index](#e_gcl()forward_with_gathered_index)
    - [E_GCL().node_model](#e_gcl()node_model)
    - [E_GCL().positions2radial](#e_gcl()positions2radial)
    - [E_GCL().positions_model](#e_gcl()positions_model)
  - [unsorted_segment_sum](#unsorted_segment_sum)

## EGNN

[Show source in EGNN.py:154](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L154)

#### Signature

```python
class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        positions_weight=1.0,
        attention=True,
        node_attr=True,
    ): ...
```

### EGNN().forward

[Show source in EGNN.py:200](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L200)

#### Signature

```python
def forward(self, x, positions, edge_index, edge_attr=None): ...
```

### EGNN().forward_with_gathered_index

[Show source in EGNN.py:216](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L216)

#### Signature

```python
def forward_with_gathered_index(
    self, gathered_x, positions, edge_index, periodic_index_mapping
): ...
```



## E_GCL

[Show source in EGNN.py:16](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L16)

#### Signature

```python
class E_GCL(nn.Module):
    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_attr_dim=0,
        act_fn=nn.ReLU(),
        positions_weight=1.0,
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=False,
        tanh=False,
    ): ...
```

### E_GCL().edge_model

[Show source in EGNN.py:72](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L72)

#### Signature

```python
def edge_model(self, source, target, radial, edge_attr): ...
```

### E_GCL().forward

[Show source in EGNN.py:116](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L116)

h: (N, emb)
positions: (N, 3)
edge_index: (2, M)
node_attr: None or (N, node_input_dim), where node_input_dim=1
edge_attr: None or (M, edge_input_dim)

#### Signature

```python
def forward(self, h, positions, edge_index, node_attr=None, edge_attr=None): ...
```

### E_GCL().forward_with_gathered_index

[Show source in EGNN.py:137](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L137)

#### Signature

```python
def forward_with_gathered_index(
    self, h, positions, edge_index, node_attr, periodic_index_mapping
): ...
```

### E_GCL().node_model

[Show source in EGNN.py:83](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L83)

#### Signature

```python
def node_model(self, x, edge_index, edge_attr, node_attr): ...
```

### E_GCL().positions2radial

[Show source in EGNN.py:105](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L105)

#### Signature

```python
def positions2radial(self, edge_index, positions): ...
```

### E_GCL().positions_model

[Show source in EGNN.py:95](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L95)

#### Signature

```python
def positions_model(self, positions, edge_index, positions_diff, edge_feat): ...
```



## unsorted_segment_sum

[Show source in EGNN.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/EGNN.py#L7)

Custom PyTorch op to replicate TensorFlow's [unsorted_segment_sum](#unsorted_segment_sum).

#### Signature

```python
def unsorted_segment_sum(data, segment_ids, num_segments): ...
```