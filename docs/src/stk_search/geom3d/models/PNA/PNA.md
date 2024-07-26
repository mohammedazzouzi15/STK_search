# Pna

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Pna](./index.md#pna) / Pna

> Auto-generated documentation for [src.stk_search.geom3d.models.PNA.PNA](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py) module.

- [Pna](#pna)
  - [PNA](#pna)
    - [PNA().forward](#pna()forward)
    - [PNA().get_graph_representation](#pna()get_graph_representation)
  - [PNAConv](#pnaconv)
    - [PNAConv().aggregate](#pnaconv()aggregate)
    - [PNAConv().forward](#pnaconv()forward)
    - [PNAConv().message](#pnaconv()message)
    - [PNAConv().reset_parameters](#pnaconv()reset_parameters)

## PNA

[Show source in PNA.py:128](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L128)

#### Signature

```python
class PNA(nn.Module):
    def __init__(self, num_layer, emb_dim, dropout_ratio, deg): ...
```

### PNA().forward

[Show source in PNA.py:158](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L158)

#### Signature

```python
def forward(self, *argv): ...
```

### PNA().get_graph_representation

[Show source in PNA.py:147](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L147)

#### Signature

```python
def get_graph_representation(self, batch): ...
```



## PNAConv

[Show source in PNA.py:25](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L25)

#### Signature

```python
class PNAConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg: Tensor,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        **kwargs
    ): ...
```

### PNAConv().aggregate

[Show source in PNA.py:118](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L118)

#### Signature

```python
def aggregate(
    self, inputs: Tensor, index: Tensor, dim_size: Optional[int] = None
) -> Tensor: ...
```

### PNAConv().forward

[Show source in PNA.py:87](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L87)

#### Signature

```python
def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None) -> Tensor: ...
```

### PNAConv().message

[Show source in PNA.py:103](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L103)

#### Signature

```python
def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor) -> Tensor: ...
```

### PNAConv().reset_parameters

[Show source in PNA.py:80](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PNA/PNA.py#L80)

#### Signature

```python
def reset_parameters(self): ...
```