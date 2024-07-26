# Autoencoder

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Autoencoder

> Auto-generated documentation for [src.stk_search.geom3d.models.AutoEncoder](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py) module.

- [Autoencoder](#autoencoder)
  - [AutoEncoder](#autoencoder)
    - [AutoEncoder().forward](#autoencoder()forward)
  - [VariationalAutoEncoder](#variationalautoencoder)
    - [VariationalAutoEncoder().encode](#variationalautoencoder()encode)
    - [VariationalAutoEncoder().forward](#variationalautoencoder()forward)
    - [VariationalAutoEncoder().reparameterize](#variationalautoencoder()reparameterize)
  - [L1_loss](#l1_loss)
  - [L2_loss](#l2_loss)
  - [cosine_similarity](#cosine_similarity)

## AutoEncoder

[Show source in AutoEncoder.py:32](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L32)

#### Signature

```python
class AutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1): ...
```

### AutoEncoder().forward

[Show source in AutoEncoder.py:58](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L58)

#### Signature

```python
def forward(self, x, y): ...
```



## VariationalAutoEncoder

[Show source in AutoEncoder.py:71](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L71)

#### Signature

```python
class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1): ...
```

### VariationalAutoEncoder().encode

[Show source in AutoEncoder.py:100](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L100)

#### Signature

```python
def encode(self, x): ...
```

### VariationalAutoEncoder().forward

[Show source in AutoEncoder.py:110](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L110)

#### Signature

```python
def forward(self, x, y): ...
```

### VariationalAutoEncoder().reparameterize

[Show source in AutoEncoder.py:105](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L105)

#### Signature

```python
def reparameterize(self, mu, log_var): ...
```



## L1_loss

[Show source in AutoEncoder.py:7](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L7)

#### Signature

```python
def L1_loss(p, z, average=True): ...
```



## L2_loss

[Show source in AutoEncoder.py:15](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L15)

#### Signature

```python
def L2_loss(p, z, average=True): ...
```



## cosine_similarity

[Show source in AutoEncoder.py:23](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/AutoEncoder.py#L23)

#### Signature

```python
def cosine_similarity(p, z, average=True): ...
```