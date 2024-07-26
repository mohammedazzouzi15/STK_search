# Balanced Irreps

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Segnn](./index.md#segnn) / Balanced Irreps

> Auto-generated documentation for [src.stk_search.geom3d.models.SEGNN.balanced_irreps](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/balanced_irreps.py) module.

- [Balanced Irreps](#balanced-irreps)
  - [BalancedIrreps](#balancedirreps)
  - [WeightBalancedIrreps](#weightbalancedirreps)

## BalancedIrreps

[Show source in balanced_irreps.py:5](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/balanced_irreps.py#L5)

 Allocates irreps equally along channel budget, resulting
in unequal numbers of irreps in ratios of 2l_i + 1 to 2l_j + 1.

Parameters
----------
lmax : int
    Maximum order of irreps.
vec_dim : int
    Dim of feature vector.
sh_type : bool
    if true, use spherical harmonics. Else the full set of irreps (with redundance).

Returns
-------
Irreps
    Resulting irreps for feature vectors.

#### Signature

```python
def BalancedIrreps(lmax, vec_dim, sh_type=True): ...
```



## WeightBalancedIrreps

[Show source in balanced_irreps.py:51](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/SEGNN/balanced_irreps.py#L51)

Determines an irreps_in1 type of order irreps_in2.lmax that when used in a tensor product
irreps_in1 x irreps_in2 -> irreps_in1
would have the same number of weights as for a standard linear layer, e.g. a tensor product
irreps_in1_scalar x "1x0e" -> irreps_in1_scalar

Parameters
----------
irreps_in1_scalar : o3.Irreps
    Number of hidden features, represented by zeroth order irreps.
irreps_in2 : o3.Irreps
    Irreps related to edge attributes.
sh : bool
    if true, yields equal number of every order. Else returns balanced irrep.
lmax : int
    Maximum order irreps to be considered.

Returns
-------
o3.Irreps
    Irreps for hidden feaure vectors.

#### Signature

```python
def WeightBalancedIrreps(irreps_in1_scalar, irreps_in2, sh=True, lmax=None): ...
```