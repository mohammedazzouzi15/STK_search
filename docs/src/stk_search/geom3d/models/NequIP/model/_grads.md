# Grads

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Model](./index.md#model) / Grads

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.model._grads](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_grads.py) module.

- [Grads](#grads)
  - [ForceOutput](#forceoutput)
  - [PartialForceOutput](#partialforceoutput)
  - [StressForceOutput](#stressforceoutput)

## ForceOutput

[Show source in _grads.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_grads.py#L7)

Add forces to a model that predicts energy.

#### Arguments

- `model` - the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

#### Returns

A ``GradientOutput`` wrapping ``model``.

#### Signature

```python
def ForceOutput(model: GraphModuleMixin) -> GradientOutput: ...
```



## PartialForceOutput

[Show source in _grads.py:27](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_grads.py#L27)

Add forces and partial forces to a model that predicts energy.

#### Arguments

- `model` - the energy model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

#### Returns

A ``GradientOutput`` wrapping ``model``.

#### Signature

```python
def PartialForceOutput(model: GraphModuleMixin) -> GradientOutput: ...
```



## StressForceOutput

[Show source in _grads.py:44](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_grads.py#L44)

Add forces and stresses to a model that predicts energy.

#### Arguments

- `model` - the model to wrap. Must have ``AtomicDataDict.TOTAL_ENERGY_KEY`` as an output.

#### Returns

A ``StressOutput`` wrapping ``model``.

#### Signature

```python
def StressForceOutput(model: GraphModuleMixin) -> GradientOutput: ...
```