# Initializers

[stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Gemnet](./index.md#gemnet) / Initializers

> Auto-generated documentation for [src.stk_search.geom3d.models.GemNet.initializers](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/initializers.py) module.

- [Initializers](#initializers)
  - [_standardize](#_standardize)
  - [he_orthogonal_init](#he_orthogonal_init)

## _standardize

[Show source in initializers.py:4](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/initializers.py#L4)

Makes sure that Var(W) = 1 and E[W] = 0

#### Signature

```python
def _standardize(kernel): ...
```



## he_orthogonal_init

[Show source in initializers.py:20](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GemNet/initializers.py#L20)

Generate a weight matrix with variance according to He initialization.
Based on a random (semi-)orthogonal matrix neural networks
are expected to learn better when features are decorrelated
(stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
"Dropout: a simple way to prevent neural networks from overfitting",
"Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")

#### Signature

```python
def he_orthogonal_init(tensor): ...
```