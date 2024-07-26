# Dmpnn

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Dmpnn

> Auto-generated documentation for [src.stk_search.geom3d.models.DMPNN](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DMPNN.py) module.

- [Dmpnn](#dmpnn)
  - [DMPNN](#dmpnn)
    - [DMPNN().forward](#dmpnn()forward)
  - [get_revert_edge_index](#get_revert_edge_index)

## DMPNN

[Show source in DMPNN.py:23](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DMPNN.py#L23)

#### Signature

```python
class DMPNN(nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"): ...
```

### DMPNN().forward

[Show source in DMPNN.py:45](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DMPNN.py#L45)

#### Signature

```python
def forward(self, *argv): ...
```



## get_revert_edge_index

[Show source in DMPNN.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/DMPNN.py#L13)

Corresponding to this line: https://github.com/chao1224/3D_Benchmark_dev/blob/main/Geom3D/datasets/datasets_utils.py#L90-L92

#### Signature

```python
def get_revert_edge_index(num_edge): ...
```