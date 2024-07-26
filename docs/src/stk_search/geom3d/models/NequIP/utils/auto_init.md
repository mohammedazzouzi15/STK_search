# Auto Init

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Auto Init

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils.auto_init](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/auto_init.py) module.

- [Auto Init](#auto-init)
  - [get_w_prefix](#get_w_prefix)
  - [instantiate](#instantiate)
  - [instantiate_from_cls_name](#instantiate_from_cls_name)

## get_w_prefix

[Show source in auto_init.py:246](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/auto_init.py#L246)

act as the get function and try to search for the value key from arg_dicts

#### Signature

```python
def get_w_prefix(
    key: List[str],
    arg_dicts: List[dict] = [],
    prefix: Optional[Union[str, List[str]]] = [],
    *kwargs
): ...
```



## instantiate

[Show source in auto_init.py:63](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/auto_init.py#L63)

Automatic initializing class instance by matching keys in the parameter dictionary to the constructor function.

Keys that are exactly the same, or with a 'prefix_' in all_args, optional_args will be used.
Priority:

all_args[key] < all_args[prefix_key] < optional_args[key] < optional_args[prefix_key] < positional_args

#### Arguments

- `builder` - the type of the instance
- `prefix` - the prefix used to address the parameter keys
- `positional_args` - the arguments used for input. These arguments have the top priority.
- `optional_args` - the second priority group to search for keys.
- `all_args` - the third priority group to search for keys.
- `remove_kwargs` - if True, ignore the kwargs argument in the init funciton
    same definition as the one in Config.from_function
- `return_args_only` *bool* - if True, do not instantiate, only return the arguments

#### Signature

```python
def instantiate(
    builder,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: dict = None,
    all_args: dict = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
    parent_builders: list = [],
): ...
```



## instantiate_from_cls_name

[Show source in auto_init.py:8](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/auto_init.py#L8)

Initialize a class based on a string class name

#### Arguments

- `module` - the module to import the class, i.e. torch.optim
- `class_name` - the string name of the class, i.e. "CosineAnnealingWarmRestarts"
- `positional_args` *dict* - positional arguments
optional_args (optional, dict): optional arguments
- `all_args` *dict* - list of all candidate parameters tha could potentially match the argument list
- `remove_kwargs` - if True, ignore the kwargs argument in the init funciton
    same definition as the one in Config.from_function
- `return_args_only` *bool* - if True, do not instantiate, only return the arguments

#### Returns

- `instance` - the instance
optional_args (dict):

#### Signature

```python
def instantiate_from_cls_name(
    module,
    class_name: str,
    prefix: Optional[Union[str, List[str]]] = [],
    positional_args: dict = {},
    optional_args: Optional[dict] = None,
    all_args: Optional[dict] = None,
    remove_kwargs: bool = True,
    return_args_only: bool = False,
): ...
```