# Build

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Model](./index.md#model) / Build

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.model._build](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_build.py) module.

- [Build](#build)
  - [model_from_config](#model_from_config)

## model_from_config

[Show source in _build.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_build.py#L10)

Build a model based on `config`.

Model builders (`model_builders`) can have arguments:
 - ``config``: the config. Always present.
 - ``model``: the model produced by the previous builder. Cannot be requested by the first builder, must be requested by subsequent ones.
 - ``initialize``: whether to initialize the model
 - ``dataset``: if ``initialize`` is True, the dataset

#### Arguments

config
- `initialize` *bool* - if True (default False), ``model_initializers`` will also be run.
- `dataset` - dataset for initializers if ``initialize`` is True.

#### Returns

The build model.

#### Signature

```python
def model_from_config(
    config, initialize: bool = False, dataset=None
) -> GraphModuleMixin: ...
```