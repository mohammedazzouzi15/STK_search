# Weight Init

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Model](./index.md#model) / Weight Init

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.model._weight_init](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_weight_init.py) module.

- [Weight Init](#weight-init)
  - [initialize_from_state](#initialize_from_state)
  - [load_model_state](#load_model_state)
  - [uniform_initialize_FCs](#uniform_initialize_fcs)
  - [unit_uniform_init_](#unit_uniform_init_)

## initialize_from_state

[Show source in _weight_init.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_weight_init.py#L13)

Initialize the model from the state dict file given by the config options `initial_model_state`.

Only loads the state dict if `initialize` is `True`; this is meant for, say, starting a training from a previous state.

If `initial_model_state_strict` controls
> whether to strictly enforce that the keys in state_dict
> match the keys returned by this module's state_dict() function

See https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict.

#### Signature

```python
def initialize_from_state(config: Config, model: GraphModuleMixin, initialize: bool): ...
```



## load_model_state

[Show source in _weight_init.py:31](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_weight_init.py#L31)

Load the model from the state dict file given by the config options [load_model_state](#load_model_state).

Loads the state dict always; this is meant, for example, for building a new model to deploy with a given state dict.

If `load_model_state_strict` controls
> whether to strictly enforce that the keys in state_dict
> match the keys returned by this module's state_dict() function

See https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict.

#### Signature

```python
def load_model_state(
    config: Config,
    model: GraphModuleMixin,
    initialize: bool,
    _prefix: str = "load_model_state",
): ...
```



## uniform_initialize_FCs

[Show source in _weight_init.py:69](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_weight_init.py#L69)

Initialize ``e3nn.nn.FullyConnectedNet``s with unit uniform initialization

#### Signature

```python
def uniform_initialize_FCs(model: GraphModuleMixin, initialize: bool): ...
```



## unit_uniform_init_

[Show source in _weight_init.py:57](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/model/_weight_init.py#L57)

Uniform initialization with <x_i^2> = 1

#### Signature

```python
def unit_uniform_init_(t: torch.Tensor): ...
```