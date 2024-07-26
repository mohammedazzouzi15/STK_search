# Layer Norm

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Modules](./index.md#modules) / Layer Norm

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.modules.layer_norm](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/layer_norm.py) module.

- [Layer Norm](#layer-norm)
  - [FusedLayerNorm](#fusedlayernorm)
    - [FusedLayerNorm().forward](#fusedlayernorm()forward)
  - [LayerNorm](#layernorm)

## FusedLayerNorm

[Show source in layer_norm.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/layer_norm.py#L11)

#### Signature

```python
class FusedLayerNorm(_FusedLayerNorm): ...
```

### FusedLayerNorm().forward

[Show source in layer_norm.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/layer_norm.py#L12)

#### Signature

```python
@torch.jit.unused
def forward(self, x): ...
```



## LayerNorm

[Show source in layer_norm.py:24](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/layer_norm.py#L24)

#### Signature

```python
def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, export=False): ...
```