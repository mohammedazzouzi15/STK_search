# Savenload

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Savenload

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils.savenload](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py) module.

- [Savenload](#savenload)
  - [_process_moves](#_process_moves)
  - [adjust_format_name](#adjust_format_name)
  - [atomic_write](#atomic_write)
  - [atomic_write_group](#atomic_write_group)
  - [atomic_write_group](#atomic_write_group-1)
  - [finish_all_writes](#finish_all_writes)
  - [finish_all_writes](#finish_all_writes-1)
  - [load_callable](#load_callable)
  - [load_file](#load_file)
  - [match_suffix](#match_suffix)
  - [save_file](#save_file)

## _process_moves

[Show source in savenload.py:33](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L33)

blocking to copy (possibly across filesystems) to temp name; then atomic rename to final name

#### Signature

```python
def _process_moves(moves: List[Tuple[bool, Path, Path]]): ...
```



## adjust_format_name

[Show source in savenload.py:313](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L313)

Recognize whether proper suffix is added to the filename.
If not, add it and return the formatted file name

#### Arguments

- `supported_formats` *dict* - list of supported formats and corresponding suffix
- `filename` *str* - initial filename
- `enforced_format` *str* - default format

#### Returns

- `newformat` *str* - the chosen format
- `newname` *str* - the adjusted filename

#### Signature

```python
def adjust_format_name(
    supported_formats: dict, filename: str, enforced_format: str = None
): ...
```



## atomic_write

[Show source in savenload.py:149](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L149)

#### Signature

```python
@contextlib.contextmanager
def atomic_write(
    filename: Union[Path, str, List[Union[Path, str]]],
    blocking: bool = True,
    binary: bool = False,
): ...
```



## atomic_write_group

[Show source in savenload.py:97](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L97)

#### Signature

```python
@contextlib.contextmanager
def atomic_write_group(): ...
```



## atomic_write_group

[Show source in savenload.py:133](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L133)

#### Signature

```python
@contextlib.contextmanager
def atomic_write_group(): ...
```



## finish_all_writes

[Show source in savenload.py:116](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L116)

#### Signature

```python
def finish_all_writes(): ...
```



## finish_all_writes

[Show source in savenload.py:145](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L145)

#### Signature

```python
def finish_all_writes(): ...
```



## load_callable

[Show source in savenload.py:290](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L290)

Load a callable from a name, or pass through a callable.

#### Signature

```python
def load_callable(
    obj: Union[str, Callable], prefix: Optional[str] = None
) -> Callable: ...
```



## load_file

[Show source in savenload.py:248](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L248)

Load file. Current support form

#### Signature

```python
def load_file(supported_formats: dict, filename: str, enforced_format: str = None): ...
```



## match_suffix

[Show source in savenload.py:357](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L357)

Recognize format based on suffix

#### Arguments

- `supported_formats` *dict* - list of supported formats and corresponding suffix
- `filename` *str* - initial filename

#### Returns

- `format` *str* - the recognized format

#### Signature

```python
def match_suffix(supported_formats: str, filename: str): ...
```



## save_file

[Show source in savenload.py:185](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/savenload.py#L185)

Save file. It can take yaml, json, pickle, json, npz and torch save

#### Signature

```python
def save_file(
    item,
    supported_formats: dict,
    filename: str,
    enforced_format: str = None,
    blocking: bool = True,
): ...
```