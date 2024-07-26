# Atomwise

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Atomwise

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._atomwise](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py) module.

- [Atomwise](#atomwise)
  - [AtomwiseLinear](#atomwiselinear)
    - [AtomwiseLinear().forward](#atomwiselinear()forward)
  - [AtomwiseOperation](#atomwiseoperation)
    - [AtomwiseOperation().forward](#atomwiseoperation()forward)
  - [AtomwiseReduce](#atomwisereduce)
    - [AtomwiseReduce().forward](#atomwisereduce()forward)
  - [PerSpeciesScaleShift](#perspeciesscaleshift)
    - [PerSpeciesScaleShift().forward](#perspeciesscaleshift()forward)
    - [PerSpeciesScaleShift().update_for_rescale](#perspeciesscaleshift()update_for_rescale)

## AtomwiseLinear

[Show source in _atomwise.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L31)

#### Signature

```python
class AtomwiseLinear(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
        irreps_out=None,
    ): ...
```

### AtomwiseLinear().forward

[Show source in _atomwise.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L55)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## AtomwiseOperation

[Show source in _atomwise.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L15)

#### Signature

```python
class AtomwiseOperation(GraphModuleMixin, torch.nn.Module):
    def __init__(self, operation, field: str, irreps_in=None): ...
```

### AtomwiseOperation().forward

[Show source in _atomwise.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L26)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## AtomwiseReduce

[Show source in _atomwise.py:60](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L60)

#### Signature

```python
class AtomwiseReduce(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
    ): ...
```

### AtomwiseReduce().forward

[Show source in _atomwise.py:88](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L88)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## PerSpeciesScaleShift

[Show source in _atomwise.py:98](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L98)

Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

#### Arguments

- `field` - the per-atom field to scale/shift.
- `num_types` - the number of types in the model.
- `shifts` - the initial shifts to use, one per atom type.
- `scales` - the initial scales to use, one per atom type.
- `arguments_in_dataset_units` - if ``True``, says that the provided shifts/scales are in dataset
    units (in which case they will be rescaled appropriately by any global rescaling later
    applied to the model); if ``False``, the provided shifts/scales will be used without modification.

    For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
    But if scales/shifts computed from the training data are used, and are thus in dataset units,
    this should be ``True``.
- `out_field` - the output field; defaults to ``field``.

#### Signature

```python
class PerSpeciesScaleShift(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str,
        num_types: int,
        type_names: List[str],
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        arguments_in_dataset_units: bool,
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        irreps_in={},
    ): ...
```

### PerSpeciesScaleShift().forward

[Show source in _atomwise.py:173](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L173)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```

### PerSpeciesScaleShift().update_for_rescale

[Show source in _atomwise.py:190](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_atomwise.py#L190)

#### Signature

```python
def update_for_rescale(self, rescale_module): ...
```