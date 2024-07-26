# Scaling

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Model](./index.md#model) / Scaling

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.model._scaling](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_scaling.py) module.

- [Scaling](#scaling)
  - [GlobalRescale](#globalrescale)
  - [PerSpeciesRescale](#perspeciesrescale)
  - [RescaleEnergyEtc](#rescaleenergyetc)
  - [_compute_stats](#_compute_stats)

## GlobalRescale

[Show source in _scaling.py:34](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_scaling.py#L34)

Add global rescaling for energy(-based quantities).

If ``initialize`` is false, doesn't compute statistics.

#### Signature

```python
def GlobalRescale(
    model: GraphModuleMixin,
    config,
    dataset: AtomicDataset,
    initialize: bool,
    module_prefix: str,
    default_scale: Union[str, float, list],
    default_shift: Union[str, float, list],
    default_scale_keys: list,
    default_shift_keys: list,
    default_related_scale_keys: list,
    default_related_shift_keys: list,
): ...
```



## PerSpeciesRescale

[Show source in _scaling.py:129](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_scaling.py#L129)

Add global rescaling for energy(-based quantities).

If ``initialize`` is false, doesn't compute statistics.

#### Signature

```python
def PerSpeciesRescale(
    model: GraphModuleMixin, config, dataset: AtomicDataset, initialize: bool
): ...
```



## RescaleEnergyEtc

[Show source in _scaling.py:14](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_scaling.py#L14)

#### Signature

```python
def RescaleEnergyEtc(
    model: GraphModuleMixin, config, dataset: AtomicDataset, initialize: bool
): ...
```



## _compute_stats

[Show source in _scaling.py:269](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_scaling.py#L269)

return the values of statistics over dataset
quantity name should be dataset_key_stat, where key can be any key
that exists in the dataset, stat can be mean, std

#### Arguments

- `str_names` - list of strings that define the quantity to compute
- `dataset` - dataset object to run the stats over
- `stride` - # frames to skip for every one frame to include

#### Signature

```python
def _compute_stats(
    str_names: List[str], dataset, stride: int, kwargs: Optional[dict] = {}
): ...
```