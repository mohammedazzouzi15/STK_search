# Gearnet Layer

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Gearnet Layer

> Auto-generated documentation for [src.stk_search.geom3d.models.GearNet_layer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py) module.

- [Gearnet Layer](#gearnet-layer)
  - [GeometricRelationalGraphConv](#geometricrelationalgraphconv)
    - [GeometricRelationalGraphConv().aggregate](#geometricrelationalgraphconv()aggregate)
    - [GeometricRelationalGraphConv().combine](#geometricrelationalgraphconv()combine)
    - [GeometricRelationalGraphConv().forward](#geometricrelationalgraphconv()forward)
    - [GeometricRelationalGraphConv().message](#geometricrelationalgraphconv()message)
  - [IEConvLayer](#ieconvlayer)
    - [IEConvLayer().aggregate](#ieconvlayer()aggregate)
    - [IEConvLayer().combine](#ieconvlayer()combine)
    - [IEConvLayer().forward](#ieconvlayer()forward)
    - [IEConvLayer().message](#ieconvlayer()message)
  - [MultiLayerPerceptron](#multilayerperceptron)
    - [MultiLayerPerceptron().forward](#multilayerperceptron()forward)
  - [SpatialLineGraph](#spatiallinegraph)
    - [SpatialLineGraph().forward](#spatiallinegraph()forward)
  - [construct_line_graph](#construct_line_graph)

## GeometricRelationalGraphConv

[Show source in GearNet_layer.py:123](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L123)

#### Signature

```python
class GeometricRelationalGraphConv(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        num_relation,
        edge_input_dim=None,
        batch_norm=False,
        activation="relu",
    ): ...
```

### GeometricRelationalGraphConv().aggregate

[Show source in GearNet_layer.py:159](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L159)

#### Signature

```python
def aggregate(self, graph, message): ...
```

### GeometricRelationalGraphConv().combine

[Show source in GearNet_layer.py:169](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L169)

#### Signature

```python
def combine(self, input, update): ...
```

### GeometricRelationalGraphConv().forward

[Show source in GearNet_layer.py:177](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L177)

#### Signature

```python
def forward(self, graph, input, edge_input=None): ...
```

### GeometricRelationalGraphConv().message

[Show source in GearNet_layer.py:149](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L149)

#### Signature

```python
def message(self, graph, input, edge_input=None): ...
```



## IEConvLayer

[Show source in GearNet_layer.py:56](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L56)

#### Signature

```python
class IEConvLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_input_dim,
        kernel_hidden_dim=32,
        dropout=0.05,
        dropout_before_conv=0.2,
        activation="relu",
        aggregate_func="sum",
    ): ...
```

### IEConvLayer().aggregate

[Show source in GearNet_layer.py:96](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L96)

#### Signature

```python
def aggregate(self, graph, message): ...
```

### IEConvLayer().combine

[Show source in GearNet_layer.py:106](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L106)

#### Signature

```python
def combine(self, input, update): ...
```

### IEConvLayer().forward

[Show source in GearNet_layer.py:110](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L110)

#### Signature

```python
def forward(self, graph, input, edge_input): ...
```

### IEConvLayer().message

[Show source in GearNet_layer.py:86](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L86)

#### Signature

```python
def message(self, graph, input, edge_input): ...
```



## MultiLayerPerceptron

[Show source in GearNet_layer.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L9)

#### Signature

```python
class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        short_cut=False,
        batch_norm=False,
        activation="relu",
        dropout=0,
    ): ...
```

### MultiLayerPerceptron().forward

[Show source in GearNet_layer.py:37](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L37)

#### Signature

```python
def forward(self, input): ...
```



## SpatialLineGraph

[Show source in GearNet_layer.py:184](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L184)

#### Signature

```python
class SpatialLineGraph(nn.Module):
    def __init__(self, num_angle_bin=8): ...
```

### SpatialLineGraph().forward

[Show source in GearNet_layer.py:189](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L189)

Generate the spatial line graph of the input graph.
The edge types are decided by the angles between two adjacent edges in the input graph.

#### Arguments

- `graph` *PackedGraph* - :math:`n` graph(s)

#### Returns

- `graph` *PackedGraph* - the spatial line graph

#### Signature

```python
def forward(self, graph): ...
```



## construct_line_graph

[Show source in GearNet_layer.py:233](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet_layer.py#L233)

Construct a packed line graph of this packed graph.
The node features of the line graphs are inherited from the edge features of the original graphs.

In the line graph, each node corresponds to an edge in the original graph.
For a pair of edges (a, b) and (b, c) that share the same intermediate node in the original graph,
there is a directed edge (a, b) -> (b, c) in the line graph.

#### Returns

PackedGraph

#### Signature

```python
def construct_line_graph(graph): ...
```