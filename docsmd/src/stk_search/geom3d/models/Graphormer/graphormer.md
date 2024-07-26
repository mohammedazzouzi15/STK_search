# Graphormer

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Graphormer](./index.md#graphormer) / Graphormer

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.graphormer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py) module.

- [Graphormer](#graphormer)
  - [GraphormerEncoder](#graphormerencoder)
    - [GraphormerEncoder().forward](#graphormerencoder()forward)
    - [GraphormerEncoder().max_nodes](#graphormerencoder()max_nodes)
    - [GraphormerEncoder().reset_output_layer_parameters](#graphormerencoder()reset_output_layer_parameters)
    - [GraphormerEncoder().upgrade_state_dict_named](#graphormerencoder()upgrade_state_dict_named)
  - [base_architecture](#base_architecture)
  - [graphormer_base_architecture](#graphormer_base_architecture)
  - [graphormer_large_architecture](#graphormer_large_architecture)
  - [graphormer_slim_architecture](#graphormer_slim_architecture)

## GraphormerEncoder

[Show source in graphormer.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L17)

#### Signature

```python
class GraphormerEncoder(nn.Module):
    def __init__(self, args): ...
```

### GraphormerEncoder().forward

[Show source in graphormer.py:79](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L79)

#### Signature

```python
def forward(self, batched_data, perturb=None, masked_tokens=None, **unused): ...
```

### GraphormerEncoder().max_nodes

[Show source in graphormer.py:105](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L105)

Maximum output length supported by the encoder.

#### Signature

```python
def max_nodes(self): ...
```

### GraphormerEncoder().reset_output_layer_parameters

[Show source in graphormer.py:74](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L74)

#### Signature

```python
def reset_output_layer_parameters(self): ...
```

### GraphormerEncoder().upgrade_state_dict_named

[Show source in graphormer.py:109](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L109)

#### Signature

```python
def upgrade_state_dict_named(self, state_dict, name): ...
```



## base_architecture

[Show source in graphormer.py:118](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L118)

#### Signature

```python
def base_architecture(args): ...
```



## graphormer_base_architecture

[Show source in graphormer.py:139](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L139)

#### Signature

```python
def graphormer_base_architecture(args): ...
```



## graphormer_large_architecture

[Show source in graphormer.py:189](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L189)

#### Signature

```python
def graphormer_large_architecture(args): ...
```



## graphormer_slim_architecture

[Show source in graphormer.py:170](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer.py#L170)

#### Signature

```python
def graphormer_slim_architecture(args): ...
```