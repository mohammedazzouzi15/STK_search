# MultiheadAttention

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Layers](./index.md#layers) / MultiheadAttention

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.layers.multihead_attention](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py) module.

- [MultiheadAttention](#multiheadattention)
  - [MultiheadAttention](#multiheadattention-1)
    - [MultiheadAttention().apply_sparse_mask](#multiheadattention()apply_sparse_mask)
    - [MultiheadAttention().forward](#multiheadattention()forward)
    - [MultiheadAttention().prepare_for_onnx_export_](#multiheadattention()prepare_for_onnx_export_)
    - [MultiheadAttention().reset_parameters](#multiheadattention()reset_parameters)
    - [MultiheadAttention().upgrade_state_dict_named](#multiheadattention()upgrade_state_dict_named)

## MultiheadAttention

[Show source in multihead_attention.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py#L17)

Multi-headed attention.

See "Attention Is All You Need" for more details.

#### Signature

```python
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ): ...
```

### MultiheadAttention().apply_sparse_mask

[Show source in multihead_attention.py:222](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py#L222)

#### Signature

```python
def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int): ...
```

### MultiheadAttention().forward

[Show source in multihead_attention.py:97](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py#L97)

Input shape: Time x Batch x Channel

#### Arguments

- `key_padding_mask` *ByteTensor, optional* - mask to exclude
    keys that are pads, of shape `(batch, src_len)`, where
    padding elements are indicated by 1s.
- `need_weights` *bool, optional* - return the attention weights,
    averaged over heads (default: False).
- `attn_mask` *ByteTensor, optional* - typically used to
    implement causal attention, where the mask prevents the
    attention from looking forward in time (default: None).
- `before_softmax` *bool, optional* - return the raw attention
    weights and values before the attention softmax.
- `need_head_weights` *bool, optional* - return the attention
    weights for each head. Implies *need_weights*. Default:
    return the average attention weights over all heads.

#### Signature

```python
def forward(
    self,
    query,
    key: Optional[Tensor],
    value: Optional[Tensor],
    attn_bias: Optional[Tensor],
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    before_softmax: bool = False,
    need_head_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]: ...
```

### MultiheadAttention().prepare_for_onnx_export_

[Show source in multihead_attention.py:78](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py#L78)

#### Signature

```python
def prepare_for_onnx_export_(self): ...
```

### MultiheadAttention().reset_parameters

[Show source in multihead_attention.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py#L81)

#### Signature

```python
def reset_parameters(self): ...
```

### MultiheadAttention().upgrade_state_dict_named

[Show source in multihead_attention.py:225](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/multihead_attention.py#L225)

#### Signature

```python
def upgrade_state_dict_named(self, state_dict, name): ...
```