# Bert

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Bert

> Auto-generated documentation for [src.stk_search.geom3d.models.BERT](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/BERT.py) module.

- [Bert](#bert)
  - [BertForSequenceRegression](#bertforsequenceregression)
    - [BertForSequenceRegression().forward](#bertforsequenceregression()forward)

## BertForSequenceRegression

[Show source in BERT.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/BERT.py#L7)

#### Signature

```python
class BertForSequenceRegression(BertPreTrainedModel):
    def __init__(self, config): ...
```

### BertForSequenceRegression().forward

[Show source in BERT.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/BERT.py#L15)

#### Signature

```python
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
): ...
```