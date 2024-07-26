# TransformerMEncoder

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Transformerm](./index.md#transformerm) / TransformerMEncoder

> Auto-generated documentation for [src.stk_search.geom3d.models.TransformerM.transformer_m_encoder](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m_encoder.py) module.

- [TransformerMEncoder](#transformermencoder)
  - [TransformerMEncoder](#transformermencoder-1)
    - [TransformerMEncoder().build_transformer_m_encoder_layer](#transformermencoder()build_transformer_m_encoder_layer)
    - [TransformerMEncoder().forward](#transformermencoder()forward)
  - [init_params](#init_params)

## TransformerMEncoder

[Show source in transformer_m_encoder.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m_encoder.py#L38)

#### Signature

```python
class TransformerMEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        encoder_normalize_before: bool = False,
        apply_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        sandwich_ln: bool = False,
        droppath_prob: float = 0.0,
        add_3d: bool = False,
        num_3d_bias_kernel: int = 128,
        no_2d: bool = False,
        mode_prob: str = "0.2,0.2,0.6",
    ) -> None: ...
```

### TransformerMEncoder().build_transformer_m_encoder_layer

[Show source in transformer_m_encoder.py:184](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m_encoder.py#L184)

#### Signature

```python
def build_transformer_m_encoder_layer(
    self,
    embedding_dim,
    ffn_embedding_dim,
    num_attention_heads,
    dropout,
    attention_dropout,
    activation_dropout,
    activation_fn,
    export,
    q_noise,
    qn_block_size,
    sandwich_ln,
    droppath_prob,
): ...
```

### TransformerMEncoder().forward

[Show source in transformer_m_encoder.py:214](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m_encoder.py#L214)

#### Signature

```python
def forward(
    self,
    batched_data,
    perturb=None,
    segment_labels: torch.Tensor = None,
    last_state_only: bool = False,
    positions: Optional[torch.Tensor] = None,
    token_embeddings: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
```



## init_params

[Show source in transformer_m_encoder.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m_encoder.py#L15)

#### Signature

```python
def init_params(module): ...
```