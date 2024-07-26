# Global Options

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Global Options

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils._global_options](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/_global_options.py) module.

- [Global Options](#global-options)
  - [_get_latest_global_options](#_get_latest_global_options)
  - [_set_global_options](#_set_global_options)

## _get_latest_global_options

[Show source in _global_options.py:21](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/_global_options.py#L21)

Get the config used latest to ``_set_global_options``.

This is useful for getting worker processes into the same state as the parent.

#### Signature

```python
def _get_latest_global_options() -> dict: ...
```



## _set_global_options

[Show source in _global_options.py:30](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/_global_options.py#L30)

Configure global options of libraries like `torch` and `e3nn` based on `config`.

#### Arguments

- `warn_on_override` - if True, will try to warn if new options are inconsistant with previously set ones.

#### Signature

```python
def _set_global_options(config, warn_on_override: bool = False) -> None: ...
```