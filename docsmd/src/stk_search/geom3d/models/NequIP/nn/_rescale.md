# Rescale

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Rescale

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._rescale](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_rescale.py) module.

- [Rescale](#rescale)
  - [RescaleOutput](#rescaleoutput)
    - [RescaleOutput().forward](#rescaleoutput()forward)
    - [RescaleOutput().get_inner_model](#rescaleoutput()get_inner_model)
    - [RescaleOutput().scale](#rescaleoutput()scale)
    - [RescaleOutput().unscale](#rescaleoutput()unscale)

## RescaleOutput

[Show source in _rescale.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_rescale.py#L12)

Wrap a model and rescale its outputs when in ``eval()`` mode.

#### Arguments

model : GraphModuleMixin
    The model whose outputs are to be rescaled.
scale_keys : list of keys, default []
    Which fields to rescale.
shift_keys : list of keys, default []
    Which fields to shift after rescaling.
- `related_scale_keys` - list of keys that could be contingent to this rescale
- `related_shift_keys` - list of keys that could be contingent to this rescale
scale_by : floating or Tensor, default 1.
    The scaling factor by which to multiply fields in ``scale``.
shift_by : floating or Tensor, default 0.
    The shift to add to fields in ``shift``.
irreps_in : dict, optional
    Extra inputs expected by this beyond those of `model`; this is only present for compatibility.

#### Signature

```python
class RescaleOutput(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        model: GraphModuleMixin,
        scale_keys: Union[Sequence[str], str] = [],
        shift_keys: Union[Sequence[str], str] = [],
        related_shift_keys: Union[Sequence[str], str] = [],
        related_scale_keys: Union[Sequence[str], str] = [],
        scale_by=None,
        shift_by=None,
        shift_trainable: bool = False,
        scale_trainable: bool = False,
        irreps_in: dict = {},
    ): ...
```

### RescaleOutput().forward

[Show source in _rescale.py:139](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_rescale.py#L139)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```

### RescaleOutput().get_inner_model

[Show source in _rescale.py:132](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_rescale.py#L132)

Get the outermost child module that is not another [RescaleOutput](#rescaleoutput)

#### Signature

```python
def get_inner_model(self): ...
```

### RescaleOutput().scale

[Show source in _rescale.py:153](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_rescale.py#L153)

Apply rescaling to ``data``, in place.

Only processes the data if the module is in ``eval()`` mode, unless ``force_process`` is ``True``.

#### Arguments

- `data` *map-like* - a dict, ``AtomicDataDict``, ``AtomicData``, ``torch_geometric.data.Batch``, or anything else dictionary-like
- `force_process` *bool* - if ``True``, scaling will be done regardless of whether the model is in train or evaluation mode.

#### Returns

``data``, modified in place

#### Signature

```python
@torch.jit.export
def scale(
    self, data: AtomicDataDict.Type, force_process: bool = False
) -> AtomicDataDict.Type: ...
```

### RescaleOutput().unscale

[Show source in _rescale.py:183](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_rescale.py#L183)

Apply the inverse of the rescaling operation to ``data``, in place.

Only processes the data if the module is in ``train()`` mode, unless ``force_process`` is ``True``.

#### Arguments

- `data` *map-like* - a dict, ``AtomicDataDict``, ``AtomicData``, ``torch_geometric.data.Batch``, or anything else dictionary-like
- `force_process` *bool* - if ``True``, unscaling will be done regardless of whether the model is in train or evaluation mode.

#### Returns

``data``

#### Signature

```python
@torch.jit.export
def unscale(
    self, data: AtomicDataDict.Type, force_process: bool = False
) -> AtomicDataDict.Type: ...
```