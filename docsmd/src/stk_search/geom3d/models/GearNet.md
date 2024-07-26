# Gearnet

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Gearnet

> Auto-generated documentation for [src.stk_search.geom3d.models.GearNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet.py) module.

- [Gearnet](#gearnet)
  - [GearNet](#gearnet)
    - [GearNet().forward](#gearnet()forward)

## GearNet

[Show source in GearNet.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet.py#L13)

#### Signature

```python
class GearNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        num_relation,
        edge_input_dim=None,
        num_angle_bin=None,
        short_cut=False,
        batch_norm=False,
        activation="relu",
        concat_hidden=False,
        readout="sum",
    ): ...
```

### GearNet().forward

[Show source in GearNet.py:53](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GearNet.py#L53)

#### Signature

```python
def forward(self, graph, input, all_loss=None, metric=None): ...
```