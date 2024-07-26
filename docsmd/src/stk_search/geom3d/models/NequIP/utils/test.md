# Test

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Test

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils.test](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/test.py) module.

- [Test](#test)
  - [assert_AtomicData_equivariant](#assert_atomicdata_equivariant)
  - [assert_permutation_equivariant](#assert_permutation_equivariant)
  - [set_irreps_debug](#set_irreps_debug)

## assert_AtomicData_equivariant

[Show source in test.py:135](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/test.py#L135)

Test the rotation, translation, parity, and permutation equivariance of ``func``.

For details on permutation testing, see [assert_permutation_equivariant](#assert_permutation_equivariant).
For details on geometric equivariance testing, see ``e3nn.util.test.assert_equivariant``.

Raises ``AssertionError`` if issues are found.

#### Arguments

- `func` - the module or model to test
- `data_in` - the example input data(s) to test with. Only the first is used for permutation testing.
- `**kwargs` - passed to ``e3nn.util.test.assert_equivariant``

#### Returns

A string description of the errors.

#### Signature

```python
def assert_AtomicData_equivariant(
    func: GraphModuleMixin,
    data_in: Union[
        AtomicData, AtomicDataDict.Type, List[Union[AtomicData, AtomicDataDict.Type]]
    ],
    permutation_tolerance: Optional[float] = None,
    o3_tolerance: Optional[float] = None,
    **kwargs
) -> str: ...
```



## assert_permutation_equivariant

[Show source in test.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/test.py#L26)

Test the permutation equivariance of ``func``.

Standard fields are assumed to be equivariant to node or edge permutations according to their standard interpretions; all other fields are assumed to be invariant to all permutations. Non-standard fields can be registered as node/edge permutation equivariant using ``register_fields``.

Raises ``AssertionError`` if issues are found.

#### Arguments

- `func` - the module or model to test
- `data_in` - the example input data to test with

#### Signature

```python
def assert_permutation_equivariant(
    func: GraphModuleMixin,
    data_in: AtomicDataDict.Type,
    tolerance: Optional[float] = None,
    raise_error: bool = True,
): ...
```



## set_irreps_debug

[Show source in test.py:310](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/test.py#L310)

Add debugging hooks to ``forward()`` that check data-irreps consistancy.

#### Arguments

- `enabled` - whether to set debug mode as enabled or disabled

#### Signature

```python
def set_irreps_debug(enabled: bool = False) -> None: ...
```