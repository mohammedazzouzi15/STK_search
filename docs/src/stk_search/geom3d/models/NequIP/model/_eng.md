# Eng

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Model](./index.md#model) / Eng

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.model._eng](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_eng.py) module.

- [Eng](#eng)
  - [EnergyModel](#energymodel)
  - [SimpleIrrepsConfig](#simpleirrepsconfig)

## EnergyModel

[Show source in _eng.py:86](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_eng.py#L86)

Base default energy model archetecture.

For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.

#### Signature

```python
def EnergyModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork: ...
```



## SimpleIrrepsConfig

[Show source in _eng.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_eng.py#L22)

Builder that pre-processes options to allow "simple" configuration of irreps.

#### Signature

```python
def SimpleIrrepsConfig(config, prefix: Optional[str] = None): ...
```