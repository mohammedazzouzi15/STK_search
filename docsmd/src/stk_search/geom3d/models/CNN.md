# Cnn

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Cnn

> Auto-generated documentation for [src.stk_search.geom3d.models.CNN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CNN.py) module.

- [Cnn](#cnn)
  - [CNN](#cnn)
    - [CNN().forward](#cnn()forward)

## CNN

[Show source in CNN.py:6](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CNN.py#L6)

#### Signature

```python
class CNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_size,
        hidden_size,
        num_tasks,
        out_channels=16,
        kernel_size=8,
        dropout=0.3,
    ): ...
```

### CNN().forward

[Show source in CNN.py:39](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/CNN.py#L39)

#### Signature

```python
def forward(self, x): ...
```