# Linear

[stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Allegro](../../index.md#allegro) / [Nn](../index.md#nn) / [Strided](./index.md#strided) / Linear

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._strided._linear](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_linear.py) module.

- [Linear](#linear)
  - [Instruction](#instruction)
  - [Linear](#linear-1)
  - [codegen_strided_linear](#codegen_strided_linear)

## Instruction

[Show source in _linear.py:16](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_linear.py#L16)

#### Signature

```python
class Instruction(NamedTuple): ...
```



## Linear

[Show source in _linear.py:202](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_linear.py#L202)

#### Signature

```python
def Linear(
    irreps_in,
    irreps_out,
    shared_weights: Optional[bool] = None,
    internal_weights: bool = False,
    instructions: Optional[List[Tuple[int, int]]] = None,
    pad_to_alignment: int = 1,
): ...
```



## codegen_strided_linear

[Show source in _linear.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_linear.py#L22)

Returns None if strided doesn't make sense for this TP.

#### Signature

```python
def codegen_strided_linear(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    normalization: str = "component",
    internal_weights: bool = False,
    shared_weights: bool = False,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]: ...
```

#### See also

- [Instruction](#instruction)