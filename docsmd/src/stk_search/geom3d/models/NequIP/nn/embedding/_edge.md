# Edge

[stk_search Index](../../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../../index.md#stk-search) / [Stk Search](../../../../../index.md#stk-search) / [Models](../../../index.md#models) / [Nequip](../../index.md#nequip) / [Nn](../index.md#nn) / [Embedding](./index.md#embedding) / Edge

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn.embedding._edge](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_edge.py) module.

- [Edge](#edge)
  - [RadialBasisEdgeEncoding](#radialbasisedgeencoding)
    - [RadialBasisEdgeEncoding().forward](#radialbasisedgeencoding()forward)
  - [SphericalHarmonicEdgeAttrs](#sphericalharmonicedgeattrs)
    - [SphericalHarmonicEdgeAttrs().forward](#sphericalharmonicedgeattrs()forward)

## RadialBasisEdgeEncoding

[Show source in _edge.py:61](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_edge.py#L61)

#### Signature

```python
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ): ...
```

### RadialBasisEdgeEncoding().forward

[Show source in _edge.py:82](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_edge.py#L82)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## SphericalHarmonicEdgeAttrs

[Show source in _edge.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_edge.py#L15)

Construct edge attrs as spherical harmonic projections of edge vectors.

Parameters follow ``e3nn.o3.spherical_harmonics``.

#### Arguments

irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
- `edge_sh_normalization` *str* - the normalization scheme to use
edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field

#### Signature

```python
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ): ...
```

### SphericalHarmonicEdgeAttrs().forward

[Show source in _edge.py:52](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/embedding/_edge.py#L52)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```