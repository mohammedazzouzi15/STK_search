# Run Md

[Stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Nequip](../index.md#nequip) / [Scripts](./index.md#scripts) / Run Md

> Auto-generated documentation for [src.stk_search.geom3d.models.NequIP.scripts.run_md](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/run_md.py) module.

- [Run Md](#run-md)
  - [main](#main)
  - [save_to_xyz](#save_to_xyz)
  - [write_ase_md_config](#write_ase_md_config)

## main

[Show source in run_md.py:74](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/run_md.py#L74)

#### Signature

```python
def main(args=None): ...
```



## save_to_xyz

[Show source in run_md.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/run_md.py#L18)

Save structure to extended xyz file.

#### Arguments

- `atoms` - ase.Atoms object to save
:param logdir, str, path/to/logging/directory
- `prefix` - str, prefix to use for storing xyz files

#### Signature

```python
def save_to_xyz(atoms, logdir, prefix=""): ...
```



## write_ase_md_config

[Show source in run_md.py:34](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/NequIP/scripts/run_md.py#L34)

Write time, positions, forces, and atomic kinetic energies to log file.

#### Arguments

- `curr_atoms` - ase.Atoms object, current system to log
- `curr_step` - int, current step / frame in MD simulation
- `dt` - float, MD time step

#### Signature

```python
def write_ase_md_config(curr_atoms, curr_step, dt): ...
```