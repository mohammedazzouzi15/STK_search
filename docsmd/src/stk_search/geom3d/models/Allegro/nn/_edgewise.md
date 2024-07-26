# Edgewise

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Allegro](../index.md#allegro) / [Nn](./index.md#nn) / Edgewise

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._edgewise](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_edgewise.py) module.

- [Edgewise](#edgewise)
  - [EdgewiseEnergySum](#edgewiseenergysum)
    - [EdgewiseEnergySum().forward](#edgewiseenergysum()forward)
  - [EdgewiseReduce](#edgewisereduce)
    - [EdgewiseReduce().forward](#edgewisereduce()forward)

## EdgewiseEnergySum

[Show source in _edgewise.py:66](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_edgewise.py#L66)

Sum edgewise energies.

Includes optional per-species-pair edgewise energy scales.

#### Signature

```python
class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},
    ): ...
```

### EdgewiseEnergySum().forward

[Show source in _edgewise.py:102](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_edgewise.py#L102)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## EdgewiseReduce

[Show source in _edgewise.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_edgewise.py#L13)

Like ``NequIP.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data.

#### Signature

```python
class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        normalize_edge_reduce: bool = True,
        avg_num_neighbors: Optional[float] = None,
        reduce="sum",
        irreps_in={},
    ): ...
```

### EdgewiseReduce().forward

[Show source in _edgewise.py:45](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_edgewise.py#L45)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```