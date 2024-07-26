# O3 Building Blocks

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Segnn](./index.md#segnn) / O3 Building Blocks

> Auto-generated documentation for [src.stk_search.geom3d.models.SEGNN.o3_building_blocks](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py) module.

- [O3 Building Blocks](#o3-building-blocks)
  - [O3TensorProduct](#o3tensorproduct)
    - [O3TensorProduct().forward](#o3tensorproduct()forward)
    - [O3TensorProduct().forward_tp_rescale_bias](#o3tensorproduct()forward_tp_rescale_bias)
    - [O3TensorProduct().tensor_product_init](#o3tensorproduct()tensor_product_init)
  - [O3TensorProductSwishGate](#o3tensorproductswishgate)
    - [O3TensorProductSwishGate().forward](#o3tensorproductswishgate()forward)

## O3TensorProduct

[Show source in o3_building_blocks.py:8](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py#L8)

#### Signature

```python
class O3TensorProduct(torch.nn.Module):
    def __init__(
        self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True
    ) -> None: ...
```

### O3TensorProduct().forward

[Show source in o3_building_blocks.py:98](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py#L98)

#### Signature

```python
def forward(self, data_in1, data_in2=None) -> torch.Tensor: ...
```

### O3TensorProduct().forward_tp_rescale_bias

[Show source in o3_building_blocks.py:83](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py#L83)

#### Signature

```python
def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor: ...
```

### O3TensorProduct().tensor_product_init

[Show source in o3_building_blocks.py:54](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py#L54)

#### Signature

```python
def tensor_product_init(self) -> None: ...
```



## O3TensorProductSwishGate

[Show source in o3_building_blocks.py:104](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py#L104)

#### Signature

```python
class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None) -> None: ...
```

#### See also

- [O3TensorProduct](#o3tensorproduct)

### O3TensorProductSwishGate().forward

[Show source in o3_building_blocks.py:123](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/o3_building_blocks.py#L123)

#### Signature

```python
def forward(self, data_in1, data_in2=None) -> torch.Tensor: ...
```