# ConvNetLayer

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / ConvNetLayer

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._convnetlayer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_convnetlayer.py) module.

- [ConvNetLayer](#convnetlayer)
  - [ConvNetLayer](#convnetlayer-1)
    - [ConvNetLayer().forward](#convnetlayer()forward)

## ConvNetLayer

[Show source in _convnetlayer.py:25](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_convnetlayer.py#L25)

#### Signature

```python
class ConvNetLayer(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        feature_irreps_hidden,
        convolution=InteractionBlock,
        convolution_kwargs: dict = {},
        num_layers: int = 3,
        resnet: bool = False,
        nonlinearity_type: str = "gate",
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp", "o": "tanh"},
        nonlinearity_gates: Dict[int, Callable] = {"e": "ssp", "o": "abs"},
    ): ...
```

### ConvNetLayer().forward

[Show source in _convnetlayer.py:156](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_convnetlayer.py#L156)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```