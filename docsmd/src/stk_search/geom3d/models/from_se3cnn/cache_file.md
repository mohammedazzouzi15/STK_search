# Cache File

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [From Se3cnn](./index.md#from-se3cnn) / Cache File

> Auto-generated documentation for [src.stk_search.geom3d.models.from_se3cnn.cache_file](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py) module.

- [Cache File](#cache-file)
  - [FileSystemMutex](#filesystemmutex)
    - [FileSystemMutex().acquire](#filesystemmutex()acquire)
    - [FileSystemMutex().release](#filesystemmutex()release)
  - [LOCK_EX](#lock_ex)
  - [cached_dirpklgz](#cached_dirpklgz)
  - [lockf](#lockf)

## FileSystemMutex

[Show source in cache_file.py:19](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py#L19)

Mutual exclusion of different **processes** using the file system

#### Signature

```python
class FileSystemMutex:
    def __init__(self, filename): ...
```

### FileSystemMutex().acquire

[Show source in cache_file.py:28](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py#L28)

Locks the mutex
if it is already locked, it waits (blocking function)

#### Signature

```python
def acquire(self): ...
```

### FileSystemMutex().release

[Show source in cache_file.py:38](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py#L38)

Unlock the mutex

#### Signature

```python
def release(self): ...
```



## LOCK_EX

[Show source in cache_file.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py#L13)

#### Signature

```python
def LOCK_EX(): ...
```



## cached_dirpklgz

[Show source in cache_file.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py#L55)

Cache a function with a directory

#### Arguments

- `dirname` - the directory path
- `maxsize` - maximum size of the RAM cache (there is no limit for the directory cache)

#### Signature

```python
def cached_dirpklgz(dirname, maxsize=128): ...
```



## lockf

[Show source in cache_file.py:16](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/from_se3cnn/cache_file.py#L16)

#### Signature

```python
def lockf(fd, operation, length=0, start=0, whence=0): ...
```