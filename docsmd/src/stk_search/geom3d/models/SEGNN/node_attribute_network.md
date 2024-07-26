# NodeAttributeNetwork

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Segnn](./index.md#segnn) / NodeAttributeNetwork

> Auto-generated documentation for [src.stk_search.geom3d.models.SEGNN.node_attribute_network](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/node_attribute_network.py) module.

- [NodeAttributeNetwork](#nodeattributenetwork)
  - [NodeAttributeNetwork](#nodeattributenetwork-1)
    - [NodeAttributeNetwork().forward](#nodeattributenetwork()forward)
    - [NodeAttributeNetwork().message](#nodeattributenetwork()message)
    - [NodeAttributeNetwork().update](#nodeattributenetwork()update)

## NodeAttributeNetwork

[Show source in node_attribute_network.py:4](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/node_attribute_network.py#L4)

Computes the node and edge attributes based on relative positions

#### Signature

```python
class NodeAttributeNetwork(MessagePassing):
    def __init__(self): ...
```

### NodeAttributeNetwork().forward

[Show source in node_attribute_network.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/node_attribute_network.py#L12)

Simply sums the edge attributes

#### Signature

```python
def forward(self, edge_index, edge_attr): ...
```

### NodeAttributeNetwork().message

[Show source in node_attribute_network.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/node_attribute_network.py#L17)

The message is the edge attribute

#### Signature

```python
def message(self, edge_attr): ...
```

### NodeAttributeNetwork().update

[Show source in node_attribute_network.py:21](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/node_attribute_network.py#L21)

The input to update is the aggregated messages, and thus the node attribute

#### Signature

```python
def update(self, node_attr): ...
```