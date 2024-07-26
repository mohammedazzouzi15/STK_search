# TensorProductRescale

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / TensorProductRescale

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.tensor_product_rescale](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py) module.

- [TensorProductRescale](#tensorproductrescale)
  - [FullyConnectedTensorProductRescale](#fullyconnectedtensorproductrescale)
  - [FullyConnectedTensorProductRescaleSwishGate](#fullyconnectedtensorproductrescaleswishgate)
    - [FullyConnectedTensorProductRescaleSwishGate().forward](#fullyconnectedtensorproductrescaleswishgate()forward)
  - [LinearRS](#linearrs)
    - [LinearRS().forward](#linearrs()forward)
  - [TensorProductRescale](#tensorproductrescale-1)
    - [TensorProductRescale().calculate_fan_in](#tensorproductrescale()calculate_fan_in)
    - [TensorProductRescale().forward](#tensorproductrescale()forward)
    - [TensorProductRescale().forward_tp_rescale_bias](#tensorproductrescale()forward_tp_rescale_bias)
    - [TensorProductRescale().init_rescale_bias](#tensorproductrescale()init_rescale_bias)
  - [irreps2gate](#irreps2gate)
  - [sort_irreps_even_first](#sort_irreps_even_first)

## FullyConnectedTensorProductRescale

[Show source in tensor_product_rescale.py:144](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L144)

#### Signature

```python
class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ): ...
```

#### See also

- [TensorProductRescale](#tensorproductrescale)



## FullyConnectedTensorProductRescaleSwishGate

[Show source in tensor_product_rescale.py:195](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L195)

#### Signature

```python
class FullyConnectedTensorProductRescaleSwishGate(FullyConnectedTensorProductRescale):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ): ...
```

#### See also

- [FullyConnectedTensorProductRescale](#fullyconnectedtensorproductrescale)

### FullyConnectedTensorProductRescaleSwishGate().forward

[Show source in tensor_product_rescale.py:218](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L218)

#### Signature

```python
def forward(self, x, y, weight=None): ...
```



## LinearRS

[Show source in tensor_product_rescale.py:165](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L165)

#### Signature

```python
class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in, irreps_out, bias=True, rescale=True): ...
```

#### See also

- [FullyConnectedTensorProductRescale](#fullyconnectedtensorproductrescale)

### LinearRS().forward

[Show source in tensor_product_rescale.py:171](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L171)

#### Signature

```python
def forward(self, x): ...
```



## TensorProductRescale

[Show source in tensor_product_rescale.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L15)

#### Signature

```python
class TensorProductRescale(torch.nn.Module):
    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        bias=True,
        rescale=True,
        internal_weights=None,
        shared_weights=None,
        normalization=None,
    ): ...
```

### TensorProductRescale().calculate_fan_in

[Show source in tensor_product_rescale.py:42](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L42)

#### Signature

```python
def calculate_fan_in(self, ins): ...
```

### TensorProductRescale().forward

[Show source in tensor_product_rescale.py:139](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L139)

#### Signature

```python
def forward(self, x, y, weight=None): ...
```

### TensorProductRescale().forward_tp_rescale_bias

[Show source in tensor_product_rescale.py:125](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L125)

#### Signature

```python
def forward_tp_rescale_bias(self, x, y, weight=None): ...
```

### TensorProductRescale().init_rescale_bias

[Show source in tensor_product_rescale.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L55)

#### Signature

```python
def init_rescale_bias(self) -> None: ...
```



## irreps2gate

[Show source in tensor_product_rescale.py:177](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L177)

#### Signature

```python
def irreps2gate(irreps): ...
```



## sort_irreps_even_first

[Show source in tensor_product_rescale.py:224](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/tensor_product_rescale.py#L224)

#### Signature

```python
def sort_irreps_even_first(irreps): ...
```