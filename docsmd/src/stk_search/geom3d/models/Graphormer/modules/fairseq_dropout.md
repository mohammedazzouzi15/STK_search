# FairseqDropout

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Modules](./index.md#modules) / FairseqDropout

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.modules.fairseq_dropout](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/fairseq_dropout.py) module.

- [FairseqDropout](#fairseqdropout)
  - [FairseqDropout](#fairseqdropout-1)
    - [FairseqDropout().forward](#fairseqdropout()forward)
    - [FairseqDropout().make_generation_fast_](#fairseqdropout()make_generation_fast_)

## FairseqDropout

[Show source in fairseq_dropout.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/fairseq_dropout.py#L7)

#### Signature

```python
class FairseqDropout(nn.Module):
    def __init__(self, p, module_name=None): ...
```

### FairseqDropout().forward

[Show source in fairseq_dropout.py:14](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/fairseq_dropout.py#L14)

#### Signature

```python
def forward(self, x, inplace: bool = False): ...
```

### FairseqDropout().make_generation_fast_

[Show source in fairseq_dropout.py:20](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/fairseq_dropout.py#L20)

#### Signature

```python
def make_generation_fast_(
    self,
    name: str,
    retain_dropout: bool = False,
    retain_dropout_modules: Optional[List[str]] = None,
    **kwargs
): ...
```