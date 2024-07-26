# Allegro

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Allegro](../index.md#allegro) / [Nn](./index.md#nn) / Allegro

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._allegro](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_allegro.py) module.

- [Allegro](#allegro)
  - [Allegro_Module](#allegro_module)
    - [Allegro_Module().forward](#allegro_module()forward)

## Allegro_Module

[Show source in _allegro.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_allegro.py#L22)

#### Attributes

- `num_layers`: `int` - saved params


#### Signature

```python
class Allegro_Module(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_types: int,
        r_max: float,
        avg_num_neighbors: Optional[float] = None,
        r_start_cos_ratio: float = 0.8,
        PolynomialCutoff_p: float = 6,
        per_layer_cutoffs: Optional[List[float]] = None,
        cutoff_type: str = "polynomial",
        field: str = AtomicDataDict.EDGE_ATTRS_KEY,
        edge_invariant_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        node_invariant_field: str = AtomicDataDict.NODE_ATTRS_KEY,
        env_embed_multiplicity: int = 32,
        embed_initial_edge: bool = True,
        linear_after_env_embed: bool = False,
        nonscalars_include_parity: bool = True,
        two_body_latent=ScalarMLPFunction,
        two_body_latent_kwargs={},
        env_embed=ScalarMLPFunction,
        env_embed_kwargs={},
        latent=ScalarMLPFunction,
        latent_kwargs={},
        latent_resnet: bool = True,
        latent_resnet_update_ratios: Optional[List[float]] = None,
        latent_resnet_update_ratios_learnable: bool = False,
        latent_out_field: Optional[str] = _keys.EDGE_FEATURES,
        pad_to_alignment: int = 1,
        sparse_mode: Optional[str] = None,
        irreps_in=None,
    ): ...
```

### Allegro_Module().forward

[Show source in _allegro.py:420](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_allegro.py#L420)

Evaluate.

#### Arguments

- `data` - AtomicDataDict.Type

#### Returns

AtomicDataDict.Type

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```