# Transformer M Layers

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Transformerm](../index.md#transformerm) / [Layers](./index.md#layers) / Transformer M Layers

> Auto-generated documentation for [src.stk_search.geom3d.models.TransformerM.layers.transformer_m_layers](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py) module.

- [Transformer M Layers](#transformer-m-layers)
  - [AtomFeature](#atomfeature)
    - [AtomFeature().forward](#atomfeature()forward)
  - [AtomTaskHead](#atomtaskhead)
    - [AtomTaskHead().forward](#atomtaskhead()forward)
  - [GaussianLayer](#gaussianlayer)
    - [GaussianLayer().forward](#gaussianlayer()forward)
  - [Molecule3DBias](#molecule3dbias)
    - [Molecule3DBias().forward](#molecule3dbias()forward)
  - [MoleculeAttnBias](#moleculeattnbias)
    - [MoleculeAttnBias().forward](#moleculeattnbias()forward)
  - [NonLinear](#nonlinear)
    - [NonLinear().forward](#nonlinear()forward)
  - [gaussian](#gaussian)
  - [init_params](#init_params)

## AtomFeature

[Show source in transformer_m_layers.py:21](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L21)

Compute atom features for each atom in the molecule.

#### Signature

```python
class AtomFeature(nn.Module):
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_in_degree,
        num_out_degree,
        hidden_dim,
        n_layers,
        no_2d=False,
    ): ...
```

### AtomFeature().forward

[Show source in transformer_m_layers.py:41](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L41)

#### Signature

```python
def forward(self, batched_data, mask_2d=None): ...
```



## AtomTaskHead

[Show source in transformer_m_layers.py:242](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L242)

#### Signature

```python
class AtomTaskHead(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int): ...
```

### AtomTaskHead().forward

[Show source in transformer_m_layers.py:263](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L263)

#### Signature

```python
def forward(self, query: Tensor, attn_bias: Tensor, delta_pos: Tensor) -> Tensor: ...
```



## GaussianLayer

[Show source in transformer_m_layers.py:204](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L204)

#### Signature

```python
class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=512 * 3): ...
```

### GaussianLayer().forward

[Show source in transformer_m_layers.py:217](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L217)

#### Signature

```python
def forward(self, x, edge_types): ...
```



## Molecule3DBias

[Show source in transformer_m_layers.py:144](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L144)

Compute 3D attention bias according to the position information for each head.

#### Signature

```python
class Molecule3DBias(nn.Module):
    def __init__(
        self, num_heads, num_edges, n_layers, embed_dim, num_kernel, no_share_rpe=False
    ): ...
```

### Molecule3DBias().forward

[Show source in transformer_m_layers.py:168](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L168)

#### Signature

```python
def forward(self, batched_data): ...
```



## MoleculeAttnBias

[Show source in transformer_m_layers.py:61](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L61)

Compute attention bias for each head.

#### Signature

```python
class MoleculeAttnBias(nn.Module):
    def __init__(
        self,
        num_heads,
        num_atoms,
        num_edges,
        num_spatial,
        num_edge_dis,
        hidden_dim,
        edge_type,
        multi_hop_max_dist,
        n_layers,
        no_2d=False,
    ): ...
```

### MoleculeAttnBias().forward

[Show source in transformer_m_layers.py:84](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L84)

#### Signature

```python
def forward(self, batched_data, mask_2d=None): ...
```



## NonLinear

[Show source in transformer_m_layers.py:226](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L226)

#### Signature

```python
class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None): ...
```

### NonLinear().forward

[Show source in transformer_m_layers.py:235](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L235)

#### Signature

```python
def forward(self, x): ...
```



## gaussian

[Show source in transformer_m_layers.py:198](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L198)

#### Signature

```python
@torch.jit.script
def gaussian(x, mean, std): ...
```



## init_params

[Show source in transformer_m_layers.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/layers/transformer_m_layers.py#L12)

#### Signature

```python
def init_params(module, n_layers): ...
```