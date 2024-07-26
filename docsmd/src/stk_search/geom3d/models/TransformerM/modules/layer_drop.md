# Layer Drop

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Transformerm](../index.md#transformerm) / [Modules](./index.md#modules) / Layer Drop

> Auto-generated documentation for [src.stk_search.geom3d.models.TransformerM.modules.layer_drop](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/modules/layer_drop.py) module.

- [Layer Drop](#layer-drop)
  - [LayerDropModuleList](#layerdropmodulelist)

## LayerDropModuleList

[Show source in layer_drop.py:6](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/modules/layer_drop.py#L6)

A LayerDrop implementation based on :class:`torch.nn.ModuleList`.
We refresh the choice of which layers to drop every time we iterate
over the LayerDropModuleList instance. During evaluation we always
iterate over all layers.
Usage

```python
layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
for layer in layers:  # this might iterate over layers 1 and 3
    x = layer(x)
for layer in layers:  # this might iterate over all layers
    x = layer(x)
for layer in layers:  # this might not iterate over any layers
    x = layer(x)
```

#### Arguments

- `p` *float* - probability of dropping out each layer
- `modules` *iterable, optional* - an iterable of modules to add

#### Signature

```python
class LayerDropModuleList(nn.ModuleList):
    def __init__(self, p, modules=None): ...
```