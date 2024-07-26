# Cutoffs

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Allegro](../index.md#allegro) / [Nn](./index.md#nn) / Cutoffs

> Auto-generated documentation for [src.stk_search.geom3d.models.Allegro.nn.cutoffs](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/cutoffs.py) module.

- [Cutoffs](#cutoffs)
  - [cosine_cutoff](#cosine_cutoff)
  - [polynomial_cutoff](#polynomial_cutoff)

## cosine_cutoff

[Show source in cutoffs.py:5](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/cutoffs.py#L5)

A piecewise cosine cutoff starting the cosine decay at r_decay_factor*r_max.

Broadcasts over r_max.

#### Signature

```python
@torch.jit.script
def cosine_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, r_start_cos_ratio: float = 0.8
): ...
```



## polynomial_cutoff

[Show source in cutoffs.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Allegro/nn/cutoffs.py#L18)

Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123

Parameters
----------
r_max : tensor
    Broadcasts over r_max.

p : int
    Power used in envelope function

#### Signature

```python
@torch.jit.script
def polynomial_cutoff(
    x: torch.Tensor, r_max: torch.Tensor, p: float = 6.0
) -> torch.Tensor: ...
```