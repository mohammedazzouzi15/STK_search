# Grad Output

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Grad Output

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._grad_output](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py) module.

- [Grad Output](#grad-output)
  - [GradientOutput](#gradientoutput)
    - [GradientOutput().forward](#gradientoutput()forward)
  - [PartialForceOutput](#partialforceoutput)
    - [PartialForceOutput().forward](#partialforceoutput()forward)
  - [StressOutput](#stressoutput)
    - [StressOutput().forward](#stressoutput()forward)

## GradientOutput

[Show source in _grad_output.py:14](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py#L14)

Wrap a model and include as an output its gradient.

#### Arguments

- `func` - the model to wrap
- `of` - the name of the output field of ``func`` to take the gradient with respect to. The field must be a single scalar (i.e. have irreps ``0e``)
- `wrt` - the input field(s) of ``func`` to take the gradient of ``of`` with regards to.
- `out_field` - the field in which to return the computed gradients. Defaults to ``f"d({of})/d({wrt})"`` for each field in ``wrt``.
- `sign` - either 1 or -1; the returned gradient is multiplied by this.

#### Signature

```python
class GradientOutput(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        func: GraphModuleMixin,
        of: str,
        wrt: Union[str, List[str]],
        out_field: Optional[List[str]] = None,
        sign: float = 1.0,
    ): ...
```

### GradientOutput().forward

[Show source in _grad_output.py:72](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py#L72)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## PartialForceOutput

[Show source in _grad_output.py:143](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py#L143)

Generate partial and total forces from an energy model.

#### Arguments

- `func` - the energy model
- `vectorize` - the vectorize option to ``torch.autograd.functional.jacobian``,
    false by default since it doesn't work well.

#### Signature

```python
class PartialForceOutput(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        func: GraphModuleMixin,
        vectorize: bool = False,
        vectorize_warnings: bool = False,
    ): ...
```

### PartialForceOutput().forward

[Show source in _grad_output.py:175](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py#L175)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## StressOutput

[Show source in _grad_output.py:204](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py#L204)

Compute stress (and forces) using autograd of an energy model.

See:
    Knuth et. al. Comput. Phys. Commun 190, 33-50, 2015
    https://pure.mpg.de/rest/items/item_2085135_9/component/file_2156800/content

#### Arguments

- `func` - the energy model to wrap
- `do_forces` - whether to compute forces as well

#### Signature

```python
class StressOutput(GraphModuleMixin, torch.nn.Module):
    def __init__(self, func: GraphModuleMixin, do_forces: bool = True): ...
```

### StressOutput().forward

[Show source in _grad_output.py:246](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_grad_output.py#L246)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```