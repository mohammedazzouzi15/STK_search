# GraphormerGraphEncoderLayer

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Layers](./index.md#layers) / GraphormerGraphEncoderLayer

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.layers.graphormer_graph_encoder_layer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_graph_encoder_layer.py) module.

- [GraphormerGraphEncoderLayer](#graphormergraphencoderlayer)
  - [GraphormerGraphEncoderLayer](#graphormergraphencoderlayer-1)
    - [GraphormerGraphEncoderLayer().build_fc1](#graphormergraphencoderlayer()build_fc1)
    - [GraphormerGraphEncoderLayer().build_fc2](#graphormergraphencoderlayer()build_fc2)
    - [GraphormerGraphEncoderLayer().build_self_attention](#graphormergraphencoderlayer()build_self_attention)
    - [GraphormerGraphEncoderLayer().forward](#graphormergraphencoderlayer()forward)

## GraphormerGraphEncoderLayer

[Show source in graphormer_graph_encoder_layer.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_graph_encoder_layer.py#L18)

#### Signature

```python
class GraphormerGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        export: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        init_fn: Callable = None,
        pre_layernorm: bool = False,
    ) -> None: ...
```

### GraphormerGraphEncoderLayer().build_fc1

[Show source in graphormer_graph_encoder_layer.py:84](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_graph_encoder_layer.py#L84)

#### Signature

```python
def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size): ...
```

### GraphormerGraphEncoderLayer().build_fc2

[Show source in graphormer_graph_encoder_layer.py:87](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_graph_encoder_layer.py#L87)

#### Signature

```python
def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size): ...
```

### GraphormerGraphEncoderLayer().build_self_attention

[Show source in graphormer_graph_encoder_layer.py:90](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_graph_encoder_layer.py#L90)

#### Signature

```python
def build_self_attention(
    self, embed_dim, num_attention_heads, dropout, self_attention, q_noise, qn_block_size
): ...
```

### GraphormerGraphEncoderLayer().forward

[Show source in graphormer_graph_encoder_layer.py:108](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/layers/graphormer_graph_encoder_layer.py#L108)

LayerNorm is applied either before or after the self-attention/ffn
modules similar to the original Transformer implementation.

#### Signature

```python
def forward(
    self,
    x: torch.Tensor,
    self_attn_bias: Optional[torch.Tensor] = None,
    self_attn_mask: Optional[torch.Tensor] = None,
    self_attn_padding_mask: Optional[torch.Tensor] = None,
): ...
```