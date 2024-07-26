# Channels

[Stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Allegro](../../index.md#allegro) / [Nn](../index.md#nn) / [Strided](./index.md#strided) / Channels

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._strided._channels](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_channels.py) module.

- [Channels](#channels)
  - [MakeWeightedChannels](#makeweightedchannels)
    - [MakeWeightedChannels().forward](#makeweightedchannels()forward)

## MakeWeightedChannels

[Show source in _channels.py:9](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_channels.py#L9)

#### Signature

```python
class MakeWeightedChannels(torch.nn.Module):
    def __init__(self, irreps_in, multiplicity_out: int, pad_to_alignment: int = 1): ...
```

### MakeWeightedChannels().forward

[Show source in _channels.py:39](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_strided/_channels.py#L39)

#### Signature

```python
def forward(self, edge_attr, weights): ...
```