# InteractionBlock

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / InteractionBlock

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._interaction_block](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_interaction_block.py) module.

- [InteractionBlock](#interactionblock)
  - [InteractionBlock](#interactionblock-1)
    - [InteractionBlock().forward](#interactionblock()forward)

## InteractionBlock

[Show source in _interaction_block.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_interaction_block.py#L17)

#### Signature

```python
class InteractionBlock(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        irreps_out,
        invariant_layers=1,
        invariant_neurons=8,
        avg_num_neighbors=None,
        use_sc=True,
        nonlinearity_scalars: Dict[int, Callable] = {"e": "ssp"},
    ) -> None: ...
```

### InteractionBlock().forward

[Show source in _interaction_block.py:145](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_interaction_block.py#L145)

Evaluate interaction Block with ResNet (self-connection).

#### Arguments

- `node_input`
- `node_attr`
- `edge_src`
- `edge_dst`
- `edge_attr`
- `edge_length_embedded`

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```