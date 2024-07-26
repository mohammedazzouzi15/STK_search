# Gps

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Gps

> Auto-generated documentation for [src.stk_search.geom3d.models.GPS](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS.py) module.

- [Gps](#gps)
  - [GPSModel](#gpsmodel)
    - [GPSModel().forward](#gpsmodel()forward)
  - [SANGraphHead](#sangraphhead)
    - [SANGraphHead().forward](#sangraphhead()forward)

## GPSModel

[Show source in GPS.py:36](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS.py#L36)

#### Signature

```python
class GPSModel(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        num_tasks,
        gt_layers=5,
        gt_dim_hidden=300,
        gt_n_heads=4,
        gt_dropout=0,
        gt_attn_dropout=0.5,
        gt_layer_norm=False,
        gt_batch_norm=True,
    ): ...
```

### GPSModel().forward

[Show source in GPS.py:67](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS.py#L67)

#### Signature

```python
def forward(self, batch): ...
```



## SANGraphHead

[Show source in GPS.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS.py#L10)

#### Signature

```python
class SANGraphHead(nn.Module):
    def __init__(self, dim_in, dim_out, L=2): ...
```

### SANGraphHead().forward

[Show source in GPS.py:25](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS.py#L25)

#### Signature

```python
def forward(self, batch): ...
```