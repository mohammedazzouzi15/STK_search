# Graph Mixin

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Nn](./index.md#nn) / Graph Mixin

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.nn._graph_mixin](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py) module.

- [Graph Mixin](#graph-mixin)
  - [GraphModuleMixin](#graphmodulemixin)
    - [GraphModuleMixin()._add_independent_irreps](#graphmodulemixin()_add_independent_irreps)
    - [GraphModuleMixin()._init_irreps](#graphmodulemixin()_init_irreps)
  - [SequentialGraphNetwork](#sequentialgraphnetwork)
    - [SequentialGraphNetwork().append](#sequentialgraphnetwork()append)
    - [SequentialGraphNetwork().append_from_parameters](#sequentialgraphnetwork()append_from_parameters)
    - [SequentialGraphNetwork().forward](#sequentialgraphnetwork()forward)
    - [SequentialGraphNetwork.from_parameters](#sequentialgraphnetworkfrom_parameters)
    - [SequentialGraphNetwork().insert](#sequentialgraphnetwork()insert)
    - [SequentialGraphNetwork().insert_from_parameters](#sequentialgraphnetwork()insert_from_parameters)

## GraphModuleMixin

[Show source in _graph_mixin.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L13)

Mixin parent class for ``torch.nn.Module``s that act on and return ``AtomicDataDict.Type`` graph data.

All such classes should call ``_init_irreps`` in their ``__init__`` functions with information on the data fields they expect, require, and produce, as well as their corresponding irreps.

#### Signature

```python
class GraphModuleMixin: ...
```

### GraphModuleMixin()._add_independent_irreps

[Show source in _graph_mixin.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L81)

Insert some independent irreps that need to be exposed to the self.irreps_in and self.irreps_out.
The terms that have already appeared in the irreps_in will be removed.

#### Arguments

- `irreps` *dict* - maps names of all new fields

#### Signature

```python
def _add_independent_irreps(self, irreps: Dict[str, Any]): ...
```

### GraphModuleMixin()._init_irreps

[Show source in _graph_mixin.py:19](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L19)

Setup the expected data fields and their irreps for this graph module.

``None`` is a valid irreps in the context for anything that is invariant but not well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph, which are invariant but are integers, not ``0e`` scalars.

#### Arguments

- `irreps_in` *dict* - maps names of all input fields from previous modules or
    data to their corresponding irreps
- `my_irreps_in` *dict* - maps names of fields to the irreps they must have for
    this graph module. Will be checked for consistancy with ``irreps_in``
- `required_irreps_in` - sequence of names of fields that must be present in
    ``irreps_in``, but that can have any irreps.
- `irreps_out` *dict* - mapping names of fields that are modified/output by
    this graph module to their irreps.

#### Signature

```python
def _init_irreps(
    self,
    irreps_in: Dict[str, Any] = {},
    my_irreps_in: Dict[str, Any] = {},
    required_irreps_in: Sequence[str] = [],
    irreps_out: Dict[str, Any] = {},
): ...
```



## SequentialGraphNetwork

[Show source in _graph_mixin.py:122](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L122)

A ``torch.nn.Sequential`` of [GraphModuleMixin](#graphmodulemixin)s.

#### Arguments

modules (list or dict of [GraphModuleMixin](#graphmodulemixin)s): the sequence of graph modules. If a list, the modules will be named ``"module0", "module1", ...``.

#### Signature

```python
class SequentialGraphNetwork(GraphModuleMixin, torch.nn.Sequential):
    def __init__(
        self, modules: Union[Sequence[GraphModuleMixin], Dict[str, GraphModuleMixin]]
    ): ...
```

#### See also

- [GraphModuleMixin](#graphmodulemixin)

### SequentialGraphNetwork().append

[Show source in _graph_mixin.py:221](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L221)

Append a module to the SequentialGraphNetwork.

#### Arguments

- `name` *str* - the name for the module
- `module` *GraphModuleMixin* - the module to append

#### Signature

```python
def append(self, name: str, module: GraphModuleMixin) -> None: ...
```

#### See also

- [GraphModuleMixin](#graphmodulemixin)

### SequentialGraphNetwork().append_from_parameters

[Show source in _graph_mixin.py:233](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L233)

Build a module from parameters and append it.

#### Arguments

- `shared_params` *dict-like* - shared parameters from which to pull when instantiating the module
- `name` *str* - the name for the module
- `builder` *callable* - a class or function to build a module
- `params` *dict, optional* - extra specific parameters for this module that take priority over those in ``shared_params``

#### Signature

```python
def append_from_parameters(
    self,
    shared_params: Mapping,
    name: str,
    builder: Callable,
    params: Dict[str, Any] = {},
) -> None: ...
```

### SequentialGraphNetwork().forward

[Show source in _graph_mixin.py:355](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L355)

#### Signature

```python
def forward(self, input: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```

### SequentialGraphNetwork.from_parameters

[Show source in _graph_mixin.py:155](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L155)

Construct a ``SequentialGraphModule`` of modules built from a shared set of parameters.

For some layer, a parameter with name ``param`` will be taken, in order of priority, from:
  1. The specific value in the parameter dictionary for that layer, if provided
  2. ``name_param`` in ``shared_params`` where ``name`` is the name of the layer
  3. ``param`` in ``shared_params``

#### Arguments

- `shared_params` *dict-like* - shared parameters from which to pull when instantiating the module
- `layers` *dict* - dictionary mapping unique names of layers to either:
      1. A callable (such as a class or function) that can be used to ``instantiate`` a module for that layer
      2. A tuple of such a callable and a dictionary mapping parameter names to values. The given dictionary of parameters will override for this layer values found in ``shared_params``.
    Options 1. and 2. can be mixed.
irreps_in (optional dict): ``irreps_in`` for the first module in the sequence.

#### Returns

The constructed SequentialGraphNetwork.

#### Signature

```python
@classmethod
def from_parameters(
    cls,
    shared_params: Mapping,
    layers: Dict[str, Union[Callable, Tuple[Callable, Dict[str, Any]]]],
    irreps_in: Optional[dict] = None,
): ...
```

### SequentialGraphNetwork().insert

[Show source in _graph_mixin.py:258](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L258)

Insert a module after the module with name ``after``.

#### Arguments

- `name` - the name of the module to insert
- `module` - the moldule to insert
- `after` - the module to insert after
- `before` - the module to insert before

#### Signature

```python
def insert(
    self,
    name: str,
    module: GraphModuleMixin,
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> None: ...
```

#### See also

- [GraphModuleMixin](#graphmodulemixin)

### SequentialGraphNetwork().insert_from_parameters

[Show source in _graph_mixin.py:315](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/nn/_graph_mixin.py#L315)

Build a module from parameters and insert it after ``after``.

#### Arguments

- `shared_params` *dict-like* - shared parameters from which to pull when instantiating the module
- `name` *str* - the name for the module
- `builder` *callable* - a class or function to build a module
- `params` *dict, optional* - extra specific parameters for this module that take priority over those in ``shared_params``
- `after` - the name of the module to insert after
- `before` - the name of the module to insert before

#### Signature

```python
def insert_from_parameters(
    self,
    shared_params: Mapping,
    name: str,
    builder: Callable,
    params: Dict[str, Any] = {},
    after: Optional[str] = None,
    before: Optional[str] = None,
) -> None: ...
```