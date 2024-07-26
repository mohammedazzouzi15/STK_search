# Graphormer Layers

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Layers](./index.md#layers) / Graphormer Layers

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.layers.graphormer_layers](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_layers.py) module.

- [Graphormer Layers](#graphormer-layers)
  - [GraphAttnBias](#graphattnbias)
    - [GraphAttnBias().forward](#graphattnbias()forward)
  - [GraphNodeFeature](#graphnodefeature)
    - [GraphNodeFeature().forward](#graphnodefeature()forward)
  - [init_params](#init_params)

## GraphAttnBias

[Show source in graphormer_layers.py:74](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_layers.py#L74)

Compute attention bias for each head.

#### Signature

```python
class GraphAttnBias(nn.Module):
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
    ): ...
```

### GraphAttnBias().forward

[Show source in graphormer_layers.py:107](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_layers.py#L107)

#### Signature

```python
def forward(self, batched_data): ...
```



## GraphNodeFeature

[Show source in graphormer_layers.py:24](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_layers.py#L24)

Compute node features for each node in the graph.

#### Signature

```python
class GraphNodeFeature(nn.Module):
    def __init__(
        self, num_heads, num_atoms, num_in_degree, num_out_degree, hidden_dim, n_layers
    ): ...
```

### GraphNodeFeature().forward

[Show source in graphormer_layers.py:47](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_layers.py#L47)

#### Signature

```python
def forward(self, batched_data): ...
```



## init_params

[Show source in graphormer_layers.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_layers.py#L15)

#### Signature

```python
def init_params(module, n_layers): ...
```