# TransformerMEncoderLayer

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Transformerm](../index.md#transformerm) / [Layers](./index.md#layers) / TransformerMEncoderLayer

> Auto-generated documentation for [src.stk_search.geom3d.models.TransformerM.layers.transformer_m_encoder_layer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_encoder_layer.py) module.

- [TransformerMEncoderLayer](#transformermencoderlayer)
  - [TransformerMEncoderLayer](#transformermencoderlayer-1)
    - [TransformerMEncoderLayer().build_fc1](#transformermencoderlayer()build_fc1)
    - [TransformerMEncoderLayer().build_fc2](#transformermencoderlayer()build_fc2)
    - [TransformerMEncoderLayer().build_self_attention](#transformermencoderlayer()build_self_attention)
    - [TransformerMEncoderLayer().forward](#transformermencoderlayer()forward)

## TransformerMEncoderLayer

[Show source in transformer_m_encoder_layer.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_encoder_layer.py#L10)

Implements a Transformer-M Encoder Layer.

#### Signature

```python
class TransformerMEncoderLayer(nn.Module):
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
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
    ) -> None: ...
```

### TransformerMEncoderLayer().build_fc1

[Show source in transformer_m_encoder_layer.py:91](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_encoder_layer.py#L91)

#### Signature

```python
def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size): ...
```

### TransformerMEncoderLayer().build_fc2

[Show source in transformer_m_encoder_layer.py:94](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_encoder_layer.py#L94)

#### Signature

```python
def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size): ...
```

### TransformerMEncoderLayer().build_self_attention

[Show source in transformer_m_encoder_layer.py:97](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_encoder_layer.py#L97)

#### Signature

```python
def build_self_attention(
    self, embed_dim, num_attention_heads, dropout, self_attention, q_noise, qn_block_size
): ...
```

### TransformerMEncoderLayer().forward

[Show source in transformer_m_encoder_layer.py:115](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_encoder_layer.py#L115)

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