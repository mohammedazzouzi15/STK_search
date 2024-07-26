# Gps Layer

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Gps Layer

> Auto-generated documentation for [src.stk_search.geom3d.models.GPS_layer](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS_layer.py) module.

- [Gps Layer](#gps-layer)
  - [GPSLayer](#gpslayer)
    - [GPSLayer()._ff_block](#gpslayer()_ff_block)
    - [GPSLayer()._sa_block](#gpslayer()_sa_block)
    - [GPSLayer().extra_repr](#gpslayer()extra_repr)
    - [GPSLayer().forward](#gpslayer()forward)

## GPSLayer

[Show source in GPS_layer.py:13](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS_layer.py#L13)

#### Signature

```python
class GPSLayer(nn.Module):
    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
    ): ...
```

### GPSLayer()._ff_block

[Show source in GPS_layer.py:156](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS_layer.py#L156)

Feed Forward block.

#### Signature

```python
def _ff_block(self, x): ...
```

### GPSLayer()._sa_block

[Show source in GPS_layer.py:147](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS_layer.py#L147)

Self-attention block.

#### Signature

```python
def _sa_block(self, x, attn_mask, key_padding_mask): ...
```

### GPSLayer().extra_repr

[Show source in GPS_layer.py:162](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS_layer.py#L162)

#### Signature

```python
def extra_repr(self): ...
```

### GPSLayer().forward

[Show source in GPS_layer.py:82](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GPS_layer.py#L82)

#### Signature

```python
def forward(self, batch): ...
```