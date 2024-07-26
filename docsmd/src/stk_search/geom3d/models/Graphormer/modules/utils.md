# Utils

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Modules](./index.md#modules) / Utils

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.modules.utils](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/utils.py) module.

- [Utils](#utils)
  - [get_activation_fn](#get_activation_fn)
  - [softmax](#softmax)

## get_activation_fn

[Show source in utils.py:14](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/utils.py#L14)

Returns the activation function corresponding to `activation`

#### Signature

```python
def get_activation_fn(activation: str) -> Callable: ...
```



## softmax

[Show source in utils.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/utils.py#L7)

#### Signature

```python
def softmax(x, dim: int, onnx_trace: bool = False): ...
```