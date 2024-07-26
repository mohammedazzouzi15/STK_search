# Geossl Pdm

[Stk_search Index](../../../../README.md#stk_search-index) / `src` / [Stk Search](../../index.md#stk-search) / [Stk Search](../../index.md#stk-search) / [Models](./index.md#models) / Geossl Pdm

> Auto-generated documentation for [src.stk_search.geom3d.models.GeoSSL_PDM](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_PDM.py) module.

- [Geossl Pdm](#geossl-pdm)
  - [GeoSSL_PDM](#geossl_pdm)
    - [GeoSSL_PDM().forward](#geossl_pdm()forward)
    - [GeoSSL_PDM.get_score_target](#geossl_pdmget_score_target)

## GeoSSL_PDM

[Show source in GeoSSL_PDM.py:26](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_PDM.py#L26)

#### Signature

```python
class GeoSSL_PDM(torch.nn.Module):
    def __init__(
        self, emb_dim, sigma_begin, sigma_end, num_noise_level, noise_type, anneal_power
    ): ...
```

### GeoSSL_PDM().forward

[Show source in GeoSSL_PDM.py:63](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_PDM.py#L63)

#### Signature

```python
def forward(
    self,
    data,
    energy,
    molecule_repr,
    pos_noise_pred,
    pos_perturbed,
    pos_target,
    debug=False,
): ...
```

### GeoSSL_PDM.get_score_target

[Show source in GeoSSL_PDM.py:41](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/GeoSSL_PDM.py#L41)

#### Signature

```python
@staticmethod
@torch.no_grad()
def get_score_target(pos_perturbed, pos_target, node2graph, noise_type): ...
```