# Contract

[stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Allegro](../../index.md#allegro) / [Nn](../index.md#nn) / [Strided](./index.md#strided) / Contract

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._strided._contract](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_contract.py) module.

- [Contract](#contract)
  - [Contracter](#contracter)
  - [codegen_strided_tensor_product_forward](#codegen_strided_tensor_product_forward)

## Contracter

[Show source in _contract.py:357](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_contract.py#L357)

#### Signature

```python
def Contracter(
    irreps_in1,
    irreps_in2,
    irreps_out,
    instructions: List[Tuple[int, int, int]],
    has_weight: bool,
    connection_mode: str,
    pad_to_alignment: int = 1,
    shared_weights: bool = False,
    sparse_mode: Optional[str] = None,
): ...
```



## codegen_strided_tensor_product_forward

[Show source in _contract.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_contract.py#L18)

Returns None if strided doesn't make sense for this TP.

#### Signature

```python
def codegen_strided_tensor_product_forward(
    irreps_in1: o3.Irreps,
    in1_var: List[float],
    irreps_in2: o3.Irreps,
    in2_var: List[float],
    irreps_out: o3.Irreps,
    out_var: List[float],
    instructions: List[Instruction],
    normalization: str = "component",
    shared_weights: bool = False,
    specialized_code: bool = True,
    sparse_mode: Optional[str] = None,
    pad_to_alignment: int = 1,
) -> Optional[fx.GraphModule]: ...
```