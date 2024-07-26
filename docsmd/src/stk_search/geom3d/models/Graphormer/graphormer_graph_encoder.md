# GraphormerGraphEncoder

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Graphormer](./index.md#graphormer) / GraphormerGraphEncoder

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.graphormer_graph_encoder](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer_graph_encoder.py) module.

- [GraphormerGraphEncoder](#graphormergraphencoder)
  - [GraphormerGraphEncoder](#graphormergraphencoder-1)
    - [GraphormerGraphEncoder().build_graphormer_graph_encoder_layer](#graphormergraphencoder()build_graphormer_graph_encoder_layer)
    - [GraphormerGraphEncoder().forward](#graphormergraphencoder()forward)
  - [init_graphormer_params](#init_graphormer_params)

## GraphormerGraphEncoder

[Show source in graphormer_graph_encoder.py:46](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer_graph_encoder.py#L46)

#### Signature

```python
class GraphormerGraphEncoder(nn.Module):
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
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None: ...
```

### GraphormerGraphEncoder().build_graphormer_graph_encoder_layer

[Show source in graphormer_graph_encoder.py:165](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer_graph_encoder.py#L165)

#### Signature

```python
def build_graphormer_graph_encoder_layer(
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
    pre_layernorm,
): ...
```

### GraphormerGraphEncoder().forward

[Show source in graphormer_graph_encoder.py:193](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer_graph_encoder.py#L193)

#### Signature

```python
def forward(
    self,
    batched_data,
    perturb=None,
    last_state_only: bool = False,
    token_embeddings: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
```



## init_graphormer_params

[Show source in graphormer_graph_encoder.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/graphormer_graph_encoder.py#L22)

Initialize the weights specific to the Graphormer Model.

#### Signature

```python
def init_graphormer_params(module): ...
```