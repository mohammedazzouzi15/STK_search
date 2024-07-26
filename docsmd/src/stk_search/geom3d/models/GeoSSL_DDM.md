# Geossl Ddm

[stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Geossl Ddm

> Auto-generated documentation for [src.stk_search.geom3d.models.GeoSSL_DDM](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_DDM.py) module.

- [Geossl Ddm](#geossl-ddm)
  - [GeoSSL_DDM](#geossl_ddm)
    - [GeoSSL_DDM().forward](#geossl_ddm()forward)
  - [MultiLayerPerceptron](#multilayerperceptron)
    - [MultiLayerPerceptron().forward](#multilayerperceptron()forward)
    - [MultiLayerPerceptron().reset_parameters](#multilayerperceptron()reset_parameters)

## GeoSSL_DDM

[Show source in GeoSSL_DDM.py:45](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_DDM.py#L45)

#### Signature

```python
class GeoSSL_DDM(torch.nn.Module):
    def __init__(
        self, emb_dim, sigma_begin, sigma_end, num_noise_level, noise_type, anneal_power
    ): ...
```

### GeoSSL_DDM().forward

[Show source in GeoSSL_DDM.py:60](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_DDM.py#L60)

#### Signature

```python
def forward(self, data, node_feature, distance): ...
```



## MultiLayerPerceptron

[Show source in GeoSSL_DDM.py:8](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_DDM.py#L8)

#### Signature

```python
class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0): ...
```

### MultiLayerPerceptron().forward

[Show source in GeoSSL_DDM.py:33](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_DDM.py#L33)

#### Signature

```python
def forward(self, input): ...
```

### MultiLayerPerceptron().reset_parameters

[Show source in GeoSSL_DDM.py:28](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_DDM.py#L28)

#### Signature

```python
def reset_parameters(self): ...
```