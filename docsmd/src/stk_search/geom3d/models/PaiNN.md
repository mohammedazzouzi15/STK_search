# Painn

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Painn

> Auto-generated documentation for [src.stk_search.geom3d.models.PaiNN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py) module.

- [Painn](#painn)
  - [PaiNN](#painn)
    - [PaiNN().create_output_layers](#painn()create_output_layers)
    - [PaiNN().forward](#painn()forward)
    - [PaiNN().forward_with_gathered_index](#painn()forward_with_gathered_index)
  - [PaiNNInteraction](#painninteraction)
    - [PaiNNInteraction().forward](#painninteraction()forward)
  - [PaiNNMixing](#painnmixing)
    - [PaiNNMixing().forward](#painnmixing()forward)

## PaiNN

[Show source in PaiNN.py:118](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L118)

PaiNN - polarizable interaction neural network

#### References

.. [#PaiNN1] Schütt, Unke, Gastegger:
   Equivariant message passing for the prediction of tensorial properties and molecular spectra.
   ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

#### Signature

```python
class PaiNN(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_rbf: int,
        cutoff: float,
        n_out: int,
        readout: str,
        gamma: float = None,
        n_out_hidden: int = None,
        n_out_layers: int = 2,
        activation: Optional[Callable] = F.silu,
        max_z: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        epsilon: float = 1e-08,
    ): ...
```

### PaiNN().create_output_layers

[Show source in PaiNN.py:205](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L205)

#### Signature

```python
def create_output_layers(self): ...
```

### PaiNN().forward

[Show source in PaiNN.py:215](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L215)

Compute atomic representations/embeddings.

#### Arguments

inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

#### Returns

- `torch.Tensor` - atom-wise representation.
list of torch.Tensor: intermediate atom-wise representations, if
return_intermediate=True was used.

#### Signature

```python
def forward(
    self,
    x,
    positions,
    radius_edge_index,
    batch,
    return_latent=False,
    return_vector=False,
): ...
```

### PaiNN().forward_with_gathered_index

[Show source in PaiNN.py:266](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L266)

#### Signature

```python
def forward_with_gathered_index(
    self,
    gathered_x,
    positions,
    radius_edge_index,
    gathered_batch,
    periodic_index_mapping,
    return_latent=False,
    return_vector=False,
): ...
```



## PaiNNInteraction

[Show source in PaiNN.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L15)

PaiNN interaction block for modeling equivariant interactions of atomistic systems.

#### Signature

```python
class PaiNNInteraction(nn.Module):
    def __init__(self, n_atom_basis: int, activation: Callable): ...
```

### PaiNNInteraction().forward

[Show source in PaiNN.py:33](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L33)

Compute interaction output.

#### Arguments

- `q` - scalar input values
- `mu` - vector input values
- `Wij` - filter
- `idx_i` - index of center atom i
- `idx_j` - index of neighbors j

#### Returns

atom features after interaction

#### Signature

```python
def forward(
    self,
    q: torch.Tensor,
    mu: torch.Tensor,
    Wij: torch.Tensor,
    dir_ij: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    n_atoms: int,
): ...
```



## PaiNNMixing

[Show source in PaiNN.py:70](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L70)

PaiNN interaction block for mixing on atom features.

#### Signature

```python
class PaiNNMixing(nn.Module):
    def __init__(
        self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-08
    ): ...
```

### PaiNNMixing().forward

[Show source in PaiNN.py:92](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/PaiNN.py#L92)

Compute intraatomic mixing.

#### Arguments

- `q` - scalar input values
- `mu` - vector input values

#### Returns

atom features after interaction

#### Signature

```python
def forward(self, q: torch.Tensor, mu: torch.Tensor): ...
```