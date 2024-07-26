# Utils

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Utils

> Auto-generated documentation for [src.stk_search.geom3d.models.utils](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/utils.py) module.

- [Utils](#utils)
  - [get_basis](#get_basis)

## get_basis

[Show source in utils.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/utils.py#L10)

Precompute the SE(3)-equivariant weight basis, W_J^lk(x)

#### Arguments

- `G` - DGL graph instance of type dgl.DGLGraph
- `max_degree` - non-negative int for degree of highest feature type

#### Returns

dict of equivariant bases. Keys are in the form 'd_in,d_out'. Values are
tensors of shape (batch_size, 1, 2*d_out+1, 1, 2*d_in+1, number_of_bases)
where the 1's will later be broadcast to the number of output and input
channels

#### Signature

```python
def get_basis(cloned_d, max_degree): ...
```