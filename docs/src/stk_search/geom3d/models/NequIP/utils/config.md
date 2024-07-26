# Config

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Utils](./index.md#utils) / Config

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.utils.config](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py) module.

- [Config](#config)
  - [Config](#config-1)
    - [Config().add_allow_list](#config()add_allow_list)
    - [Config().allow_list](#config()allow_list)
    - [Config().as_dict](#config()as_dict)
    - [Config.from_class](#configfrom_class)
    - [Config.from_dict](#configfrom_dict)
    - [Config.from_file](#configfrom_file)
    - [Config.from_function](#configfrom_function)
    - [Config().get](#config()get)
    - [Config().get_type](#config()get_type)
    - [Config().items](#config()items)
    - [Config().keys](#config()keys)
    - [Config().persist](#config()persist)
    - [Config().pop](#config()pop)
    - [Config().save](#config()save)
    - [Config().set_type](#config()set_type)
    - [Config().setdefaults](#config()setdefaults)
    - [Config().update](#config()update)
    - [Config().update_locked](#config()update_locked)
    - [Config().update_w_prefix](#config()update_w_prefix)

## Config

[Show source in config.py:45](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L45)

#### Signature

```python
class Config(object):
    def __init__(
        self,
        config: Optional[dict] = None,
        allow_list: Optional[list] = None,
        exclude_keys: Optional[list] = None,
    ): ...
```

### Config().add_allow_list

[Show source in config.py:105](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L105)

add key to allow_list

#### Signature

```python
def add_allow_list(self, keys, default_values={}): ...
```

### Config().allow_list

[Show source in config.py:114](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L114)

#### Signature

```python
def allow_list(self): ...
```

### Config().as_dict

[Show source in config.py:79](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L79)

#### Signature

```python
def as_dict(self): ...
```

### Config.from_class

[Show source in config.py:273](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L273)

return Config class instance based on init function of the input class
the instance will only allow to store init function related variables
the type hints are all set to None, so no automatic format conversion is applied

class_type: torch.module children class type, i.e. .NequIP.Nequip
remove_kwargs (optional, bool): the same as Config.from_function

#### Returns

config (Config):

#### Signature

```python
@staticmethod
def from_class(class_type, remove_kwargs: bool = False): ...
```

### Config.from_dict

[Show source in config.py:267](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L267)

#### Signature

```python
@staticmethod
def from_dict(dictionary: dict, defaults: dict = {}): ...
```

### Config.from_file

[Show source in config.py:255](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L255)

Load arguments from file

#### Signature

```python
@staticmethod
def from_file(filename: str, format: Optional[str] = None, defaults: dict = {}): ...
```

### Config.from_function

[Show source in config.py:298](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L298)

return Config class instance based on the function of the input class
the instance will only allow to store init function related variables
the type hints are all set to None, so no automatic format conversion is applied

#### Arguments

- `function` - function name
remove_kwargs (optional, bool): if True, kwargs are removed from the keys
     and the returned instance will only takes the init params of the class_type.
     if False and kwargs exists, the config only initialized with the default param values,
     but it can take any other keys

#### Returns

config (Config):

#### Signature

```python
@staticmethod
def from_function(function, remove_kwargs=False): ...
```

### Config().get

[Show source in config.py:229](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L229)

#### Signature

```python
def get(self, *args): ...
```

### Config().get_type

[Show source in config.py:85](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L85)

Get Typehint from item_types dict or previous defined value

#### Arguments

- `key` - name of the variable

#### Signature

```python
def get_type(self, key): ...
```

### Config().items

[Show source in config.py:148](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L148)

#### Signature

```python
def items(self): ...
```

### Config().keys

[Show source in config.py:73](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L73)

#### Signature

```python
def keys(self): ...
```

### Config().persist

[Show source in config.py:232](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L232)

mock wandb.config function

#### Signature

```python
def persist(self): ...
```

### Config().pop

[Show source in config.py:159](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L159)

#### Signature

```python
def pop(self, *args): ...
```

### Config().save

[Show source in config.py:244](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L244)

Print config to file.

#### Signature

```python
def save(self, filename: str, format: Optional[str] = None): ...
```

### Config().set_type

[Show source in config.py:94](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L94)

set typehint for a variable

#### Arguments

- `key` - name of the variable
- `typehint` - type of the variable

#### Signature

```python
def set_type(self, key, typehint): ...
```

### Config().setdefaults

[Show source in config.py:236](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L236)

mock wandb.config function

#### Signature

```python
def setdefaults(self, d): ...
```

### Config().update

[Show source in config.py:199](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L199)

Mock of wandb.config function

Add a dictionary of parameters to the config
The key of the parameter cannot be started as "_"

#### Arguments

- `dictionary` *dict* - dictionary of parameters and their typehint to update
- `allow_val_change` *None* - mock for wandb.config, not used.

#### Returns

- [Config().keys](#configkeys) *set* - set of keys being udpated

#### Signature

```python
def update(self, dictionary: dict, allow_val_change=None): ...
```

### Config().update_locked

[Show source in config.py:240](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L240)

mock wandb.config function

#### Signature

```python
def update_locked(self, d, user=None): ...
```

### Config().update_w_prefix

[Show source in config.py:162](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/utils/config.py#L162)

Mock of wandb.config function

Add a dictionary of parameters to the
The key of the parameter cannot be started as "_"

#### Arguments

- `dictionary` *dict* - dictionary of parameters and their typehint to update
- `allow_val_change` *None* - mock for wandb.config, not used.

#### Signature

```python
def update_w_prefix(self, dictionary: dict, prefix: str, allow_val_change=None): ...
```