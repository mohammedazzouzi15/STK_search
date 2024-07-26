# Fibers

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Fibers

> Auto-generated documentation for [src.stk_search.geom3d.models.fibers](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/fibers.py) module.

- [Fibers](#fibers)
  - [Fiber](#fiber)
    - [Fiber.combine](#fibercombine)
    - [Fiber.combine_max](#fibercombine_max)
  - [fiber2head](#fiber2head)

## Fiber

[Show source in fibers.py:11](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/fibers.py#L11)

A Handy Data Structure for Fibers

#### Signature

```python
class Fiber(object):
    def __init__(
        self,
        num_degrees: int = None,
        num_channels: int = None,
        structure: List[Tuple[int, int]] = None,
        dictionary=None,
    ): ...
```

### Fiber.combine

[Show source in fibers.py:44](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/fibers.py#L44)

#### Signature

```python
@staticmethod
def combine(f1, f2): ...
```

### Fiber.combine_max

[Show source in fibers.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/fibers.py#L55)

#### Signature

```python
@staticmethod
def combine_max(f1, f2): ...
```



## fiber2head

[Show source in fibers.py:70](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/fibers.py#L70)

#### Signature

```python
def fiber2head(F, h, structure, squeeze=False): ...
```