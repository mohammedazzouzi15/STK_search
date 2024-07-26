# Fc

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Allegro](../index.md#allegro) / [Nn](./index.md#nn) / Fc

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn._fc](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_fc.py) module.

- [Fc](#fc)
  - [ScalarMLP](#scalarmlp)
    - [ScalarMLP().forward](#scalarmlp()forward)
  - [ScalarMLPFunction](#scalarmlpfunction)
    - [ScalarMLPFunction().forward](#scalarmlpfunction()forward)

## ScalarMLP

[Show source in _fc.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_fc.py#L18)

Apply an MLP to some scalar field.

#### Signature

```python
class ScalarMLP(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "uniform",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
        field: str = AtomicDataDict.NODE_FEATURES_KEY,
        out_field: Optional[str] = None,
        irreps_in=None,
    ): ...
```

### ScalarMLP().forward

[Show source in _fc.py:59](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_fc.py#L59)

#### Signature

```python
def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type: ...
```



## ScalarMLPFunction

[Show source in _fc.py:64](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_fc.py#L64)

Module implementing an MLP according to provided options.

#### Signature

```python
class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "normal",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
    ): ...
```

### ScalarMLPFunction().forward

[Show source in _fc.py:168](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/_fc.py#L168)

#### Signature

```python
def forward(self, x): ...
```