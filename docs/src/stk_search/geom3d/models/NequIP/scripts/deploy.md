# Deploy

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Scripts](./index.md#scripts) / Deploy

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.scripts.deploy](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/deploy.py) module.

- [Deploy](#deploy)
  - [load_deployed_model](#load_deployed_model)
  - [main](#main)

## load_deployed_model

[Show source in deploy.py:65](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/deploy.py#L65)

Load a deployed model.

#### Arguments

- `model_path` - the path to the deployed model's ``.pth`` file.

#### Returns

model, metadata dictionary

#### Signature

```python
def load_deployed_model(
    model_path: Union[pathlib.Path, str],
    device: Union[str, torch.device] = "cpu",
    freeze: bool = True,
    set_global_options: Union[str, bool] = "warn",
) -> Tuple[torch.jit.ScriptModule, Dict[str, str]]: ...
```



## main

[Show source in deploy.py:130](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/deploy.py#L130)

#### Signature

```python
def main(args=None): ...
```