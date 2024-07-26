# Equiformer Type 01

[Stk_search Index](../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../index.md#stk-search) / [Stk Search](../../../index.md#stk-search) / [Models](../index.md#models) / [Equiformer](./index.md#equiformer) / Equiformer Type 01

> Auto-generated documentation for [src.stk_search.geom3d.models.Equiformer.equiformer_type_01](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py) module.

- [Equiformer Type 01](#equiformer-type-01)
  - [CosineCutoff](#cosinecutoff)
    - [CosineCutoff().forward](#cosinecutoff()forward)
  - [EquiformerEnergyForce](#equiformerenergyforce)
    - [EquiformerEnergyForce().build_blocks](#equiformerenergyforce()build_blocks)
    - [EquiformerEnergyForce().forward](#equiformerenergyforce()forward)
    - [EquiformerEnergyForce().no_weight_decay](#equiformerenergyforce()no_weight_decay)
  - [ExpNormalSmearing](#expnormalsmearing)
    - [ExpNormalSmearing().forward](#expnormalsmearing()forward)
    - [ExpNormalSmearing().reset_parameters](#expnormalsmearing()reset_parameters)
  - [Equiformer_l2_energy_force](#equiformer_l2_energy_force)
  - [Equiformer_nonlinear_attn_exp_l3_energy_force](#equiformer_nonlinear_attn_exp_l3_energy_force)
  - [Equiformer_nonlinear_bessel_l2_energy_force](#equiformer_nonlinear_bessel_l2_energy_force)
  - [Equiformer_nonlinear_bessel_l3_e3_energy_force](#equiformer_nonlinear_bessel_l3_e3_energy_force)
  - [Equiformer_nonlinear_bessel_l3_energy_force](#equiformer_nonlinear_bessel_l3_energy_force)
  - [Equiformer_nonlinear_exp_l2_energy_force](#equiformer_nonlinear_exp_l2_energy_force)
  - [Equiformer_nonlinear_exp_l3_e3_energy_force](#equiformer_nonlinear_exp_l3_e3_energy_force)
  - [Equiformer_nonlinear_exp_l3_energy_force](#equiformer_nonlinear_exp_l3_energy_force)
  - [Equiformer_nonlinear_l2_e3_energy_force](#equiformer_nonlinear_l2_e3_energy_force)
  - [Equiformer_nonlinear_l2_energy_force](#equiformer_nonlinear_l2_energy_force)

## CosineCutoff

[Show source in equiformer_type_01.py:49](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L49)

#### Signature

```python
class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0): ...
```

### CosineCutoff().forward

[Show source in equiformer_type_01.py:55](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L55)

#### Signature

```python
def forward(self, distances): ...
```



## EquiformerEnergyForce

[Show source in equiformer_type_01.py:125](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L125)

#### Signature

```python
class EquiformerEnergyForce(torch.nn.Module):
    def __init__(
        self,
        irreps_in="64x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        node_class=119,
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=[64, 64],
        irreps_feature="512x0e",
        irreps_head="32x0e+16x1o+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="128x0e+64x1e+32x2e",
        use_attn_head=False,
        norm_layer="layer",
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
    ): ...
```

### EquiformerEnergyForce().build_blocks

[Show source in equiformer_type_01.py:216](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L216)

#### Signature

```python
def build_blocks(self): ...
```

### EquiformerEnergyForce().forward

[Show source in equiformer_type_01.py:273](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L273)

#### Signature

```python
@torch.enable_grad()
def forward(self, node_atom, pos, batch): ...
```

### EquiformerEnergyForce().no_weight_decay

[Show source in equiformer_type_01.py:249](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L249)

#### Signature

```python
@torch.jit.ignore
def no_weight_decay(self): ...
```



## ExpNormalSmearing

[Show source in equiformer_type_01.py:81](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L81)

#### Signature

```python
class ExpNormalSmearing(torch.nn.Module):
    def __init__(
        self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=False
    ): ...
```

### ExpNormalSmearing().forward

[Show source in equiformer_type_01.py:117](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L117)

#### Signature

```python
def forward(self, dist): ...
```

### ExpNormalSmearing().reset_parameters

[Show source in equiformer_type_01.py:112](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L112)

#### Signature

```python
def reset_parameters(self): ...
```



## Equiformer_l2_energy_force

[Show source in equiformer_type_01.py:326](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L326)

#### Signature

```python
def Equiformer_l2_energy_force(irreps_in, radius, num_basis, node_class, **kwargs): ...
```



## Equiformer_nonlinear_attn_exp_l3_energy_force

[Show source in equiformer_type_01.py:422](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L422)

#### Signature

```python
def Equiformer_nonlinear_attn_exp_l3_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_bessel_l2_energy_force

[Show source in equiformer_type_01.py:374](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L374)

#### Signature

```python
def Equiformer_nonlinear_bessel_l2_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_bessel_l3_e3_energy_force

[Show source in equiformer_type_01.py:471](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L471)

#### Signature

```python
def Equiformer_nonlinear_bessel_l3_e3_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_bessel_l3_energy_force

[Show source in equiformer_type_01.py:455](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L455)

#### Signature

```python
def Equiformer_nonlinear_bessel_l3_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_exp_l2_energy_force

[Show source in equiformer_type_01.py:390](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L390)

#### Signature

```python
def Equiformer_nonlinear_exp_l2_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_exp_l3_e3_energy_force

[Show source in equiformer_type_01.py:439](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L439)

#### Signature

```python
def Equiformer_nonlinear_exp_l3_e3_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_exp_l3_energy_force

[Show source in equiformer_type_01.py:406](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L406)

#### Signature

```python
def Equiformer_nonlinear_exp_l3_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_l2_e3_energy_force

[Show source in equiformer_type_01.py:358](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L358)

#### Signature

```python
def Equiformer_nonlinear_l2_e3_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```



## Equiformer_nonlinear_l2_energy_force

[Show source in equiformer_type_01.py:342](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Equiformer/equiformer_type_01.py#L342)

#### Signature

```python
def Equiformer_nonlinear_l2_energy_force(
    irreps_in, radius, num_basis, node_class, **kwargs
): ...
```