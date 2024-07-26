# TransformerM

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Transformerm](./index.md#transformerm) / TransformerM

> Auto-generated documentation for [src.stk_search.geom3d.models.TransformerM.transformer_m](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py) module.

- [TransformerM](#transformerm)
  - [TransformerM](#transformerm-1)
    - [TransformerM().forward](#transformerm()forward)
    - [TransformerM().max_positions](#transformerm()max_positions)
    - [TransformerM().upgrade_state_dict_named](#transformerm()upgrade_state_dict_named)
  - [base_architecture](#base_architecture)
  - [bert_base_architecture](#bert_base_architecture)

## TransformerM

[Show source in transformer_m.py:8](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py#L8)

#### Signature

```python
class TransformerM(nn.Module):
    def __init__(self, args): ...
```

### TransformerM().forward

[Show source in transformer_m.py:60](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py#L60)

#### Signature

```python
def forward(
    self, batched_data, perturb=None, segment_labels=None, masked_tokens=None, **unused
): ...
```

### TransformerM().max_positions

[Show source in transformer_m.py:76](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py#L76)

Maximum output length supported by the encoder.

#### Signature

```python
def max_positions(self): ...
```

### TransformerM().upgrade_state_dict_named

[Show source in transformer_m.py:80](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py#L80)

#### Signature

```python
def upgrade_state_dict_named(self, state_dict, name): ...
```



## base_architecture

[Show source in transformer_m.py:84](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py#L84)

#### Signature

```python
def base_architecture(args): ...
```



## bert_base_architecture

[Show source in transformer_m.py:117](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/TransformerM/transformer_m.py#L117)

#### Signature

```python
def bert_base_architecture(args): ...
```