# Gemnet

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Gemnet](./index.md#gemnet) / Gemnet

> Auto-generated documentation for [src.stk_search.geom3d.models.GemNet.GemNet](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py) module.

- [Gemnet](#gemnet)
  - [GemNet](#gemnet)
    - [GemNet.calculate_angles](#gemnetcalculate_angles)
    - [GemNet.calculate_angles3](#gemnetcalculate_angles3)
    - [GemNet.calculate_interatomic_vectors](#gemnetcalculate_interatomic_vectors)
    - [GemNet.calculate_neighbor_angles](#gemnetcalculate_neighbor_angles)
    - [GemNet().forward](#gemnet()forward)
    - [GemNet().forward_with_gathered_index](#gemnet()forward_with_gathered_index)
    - [GemNet().load_weights](#gemnet()load_weights)
    - [GemNet().predict](#gemnet()predict)
    - [GemNet().save_weights](#gemnet()save_weights)
    - [GemNet.vector_rejection](#gemnetvector_rejection)

## GemNet

[Show source in GemNet.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L26)

Parameters
----------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    num_blocks: int
        Number of building blocks to be stacked.
    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_trip: int
        (Down-projected) Embedding size in the triplet message passing block.
    emb_size_quad: int
        (Down-projected) Embedding size in the quadruplet message passing block.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    emb_size_sbf: int
        Embedding size of the spherical basis transformation (two angles).
    emb_size_bil_trip: int
        Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.
    emb_size_bil_quad: int
        Embedding size of the edge embeddings in the quadruplet-based message passing block after the bilinear layer.
    num_before_skip: int
        Number of residual blocks before the first skip connection.
    num_after_skip: int
        Number of residual blocks after the first skip connection.
    num_concat: int
        Number of residual blocks after the concatenation.
    num_atom: int
        Number of residual blocks in the atom embedding blocks.
    direct_forces: bool
        If True predict forces based on aggregation of interatomic directions.
        If False predict forces based on negative gradient of energy potential.
    triplets_only: bool
        If True use GemNet-T or GemNet-dT.No quadruplet based message passing.
    num_targets: int
        Number of prediction targets.
    cutoff: float
        Embedding cutoff for interactomic directions in Angstrom.
    int_cutoff: float
        Interaction cutoff for interactomic directions in Angstrom. No effect for GemNet-(d)T
    envelope_exponent: int
        Exponent of the envelope function. Determines the shape of the smooth cutoff.
    extensive: bool
        Whether the output should be extensive (proportional to the number of atoms)
    forces_coupled: bool
        No effect if direct_forces is False. If True enforce that |F_ac| = |F_ca|
    output_init: str
        Initialization method for the final dense layer.
    activation: str
        Name of the activation function.
    scale_file: str
        Path to the json file containing the scaling factors.

#### Signature

```python
class GemNet(torch.nn.Module):
    def __init__(
        self,
        node_class: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_quad: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        emb_size_bil_quad: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        triplets_only: bool,
        num_targets: int = 1,
        direct_forces: bool = False,
        cutoff: float = 5.0,
        int_cutoff: float = 10.0,
        envelope_exponent: int = 5,
        extensive=True,
        forces_coupled: bool = False,
        output_init="HeOrthogonal",
        activation: str = "swish",
        scale_file=None,
        name="GemNet",
        **kwargs
    ): ...
```

### GemNet.calculate_angles

[Show source in GemNet.py:340](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L340)

Calculate angles for quadruplet-based message passing.

Parameters
----------
    R: Tensor, shape = (nAtoms,3)
        Atom positions.
    id_c: Tensor, shape = (nEdges,)
        Indices of atom c (source atom of edge).
    id_a: Tensor, shape = (nEdges,)
        Indices of atom a (target atom of edge).
    id4_int_b: torch.Tensor, shape (nInterEdges,)
        Indices of the atom b of the interaction edge.
    id4_int_a: torch.Tensor, shape (nInterEdges,)
        Indices of the atom a of the interaction edge.
    id4_expand_abd: torch.Tensor, shape (nQuadruplets,)
        Indices to map from intermediate d->b to quadruplet d->b.
    id4_reduce_cab: torch.Tensor, shape (nQuadruplets,)
        Indices to map from intermediate c->a to quadruplet c->a.
    id4_expand_intm_db: torch.Tensor, shape (intmTriplets,)
        Indices to map d->b to intermediate d->b.
    id4_reduce_intm_ca: torch.Tensor, shape (intmTriplets,)
        Indices to map c->a to intermediate c->a.
    id4_expand_intm_ab: torch.Tensor, shape (intmTriplets,)
        Indices to map b-a to intermediate b-a of the quadruplet's part a-b<-d.
    id4_reduce_intm_ab: torch.Tensor, shape (intmTriplets,)
        Indices to map b-a to intermediate b-a of the quadruplet's part c->a-b.

Returns
-------
    angle_cab: Tensor, shape = (nQuadruplets,)
        Angle between atoms c <- a -> b.
    angle_abd: Tensor, shape = (intmTriplets,)
        Angle between atoms a <- b -> d.
    angle_cabd: Tensor, shape = (nQuadruplets,)
        Angle between atoms c <- a-b -> d.

#### Signature

```python
@staticmethod
def calculate_angles(
    R,
    id_c,
    id_a,
    id4_int_b,
    id4_int_a,
    id4_expand_abd,
    id4_reduce_cab,
    id4_expand_intm_db,
    id4_reduce_intm_ca,
    id4_expand_intm_ab,
    id4_reduce_intm_ab,
): ...
```

### GemNet.calculate_angles3

[Show source in GemNet.py:426](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L426)

Calculate angles for triplet-based message passing.

Parameters
----------
    R: Tensor, shape = (nAtoms,3)
        Atom positions.
    id_c: Tensor, shape = (nEdges,)
        Indices of atom c (source atom of edge).
    id_a: Tensor, shape = (nEdges,)
        Indices of atom a (target atom of edge).
    id3_reduce_ca: Tensor, shape = (nTriplets,)
        Edge indices of edge c -> a of the triplets.
    id3_expand_ba: Tensor, shape = (nTriplets,)
        Edge indices of edge b -> a of the triplets.

Returns
-------
    angle_cab: Tensor, shape = (nTriplets,)
        Angle between atoms c <- a -> b.

#### Signature

```python
@staticmethod
def calculate_angles3(R, id_c, id_a, id3_reduce_ca, id3_expand_ba): ...
```

### GemNet.calculate_interatomic_vectors

[Show source in GemNet.py:267](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L267)

Parameters
----------
    R: Tensor, shape = (nAtoms,3)
        Atom positions.
    id_s: Tensor, shape = (nEdges,)
        Indices of the source atom of the edges.
    id_t: Tensor, shape = (nEdges,)
        Indices of the target atom of the edges.

Returns
-------
    (D_st, V_st): tuple
        D_st: Tensor, shape = (nEdges,)
            Distance from atom t to s.
        V_st: Tensor, shape = (nEdges,)
            Unit direction from atom t to s.

#### Signature

```python
@staticmethod
def calculate_interatomic_vectors(R, id_s, id_t): ...
```

### GemNet.calculate_neighbor_angles

[Show source in GemNet.py:294](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L294)

Calculate angles between atoms c <- a -> b.

Parameters
----------
    R_ac: Tensor, shape = (N,3)
        Vector from atom a to c.
    R_ab: Tensor, shape = (N,3)
        Vector from atom a to b.

Returns
-------
    angle_cab: Tensor, shape = (N,)
        Angle between atoms c <- a -> b.

#### Signature

```python
@staticmethod
def calculate_neighbor_angles(R_ac, R_ab): ...
```

### GemNet().forward

[Show source in GemNet.py:459](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L459)

#### Signature

```python
def forward(self, z, positions, inputs): ...
```

### GemNet().forward_with_gathered_index

[Show source in GemNet.py:591](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L591)

#### Signature

```python
def forward_with_gathered_index(
    self, z, positions, inputs, batch, periodic_index_mapping
): ...
```

### GemNet().load_weights

[Show source in GemNet.py:725](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L725)

#### Signature

```python
def load_weights(self, path): ...
```

### GemNet().predict

[Show source in GemNet.py:719](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L719)

#### Signature

```python
def predict(self, inputs): ...
```

### GemNet().save_weights

[Show source in GemNet.py:728](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L728)

#### Signature

```python
def save_weights(self, path): ...
```

### GemNet.vector_rejection

[Show source in GemNet.py:319](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/GemNet.py#L319)

Project the vector R_ab onto a plane with normal vector P_n.

Parameters
----------
    R_ab: Tensor, shape = (N,3)
        Vector from atom a to b.
    P_n: Tensor, shape = (N,3)
        Normal vector of a plane onto which to project R_ab.

Returns
-------
    R_ab_proj: Tensor, shape = (N,3)
        Projected vector (orthogonal to P_n).

#### Signature

```python
@staticmethod
def vector_rejection(R_ab, P_n): ...
```