# Objective Function

[stk_search Index](../../README.md#stk_search-index) / `src` / [Stk Search](./index.md#stk-search) / Objective Function

> Auto-generated documentation for [src.stk_search.Objective_function](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py) module.

- [Objective Function](#objective-function)
  - [IP_ES1_fosc](#ip_es1_fosc)
    - [IP_ES1_fosc().Build_polymer](#ip_es1_fosc()build_polymer)
    - [IP_ES1_fosc().evaluate_element](#ip_es1_fosc()evaluate_element)
    - [IP_ES1_fosc().run_stda](#ip_es1_fosc()run_stda)
    - [IP_ES1_fosc().run_xtb_ipea](#ip_es1_fosc()run_xtb_ipea)
    - [IP_ES1_fosc().run_xtb_opt](#ip_es1_fosc()run_xtb_opt)
  - [Look_up_table](#look_up_table)
    - [Look_up_table().evaluate_element](#look_up_table()evaluate_element)
  - [Objective_Function](#objective_function)
    - [Objective_Function().evaluate_element](#objective_function()evaluate_element)
  - [get_inchi_key](#get_inchi_key)

## IP_ES1_fosc

[Show source in Objective_function.py:61](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L61)

#### Signature

```python
class IP_ES1_fosc(Objective_Function):
    def __init__(self, oligomer_size): ...
```

#### See also

- [Objective_Function](#objective_function)

### IP_ES1_fosc().Build_polymer

[Show source in Objective_function.py:161](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L161)

#### Signature

```python
def Build_polymer(self, element: pd.DataFrame, db: stk.MoleculeMongoDb = None): ...
```

### IP_ES1_fosc().evaluate_element

[Show source in Objective_function.py:85](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L85)

#### Signature

```python
def evaluate_element(self, element, multiFidelity=False): ...
```

### IP_ES1_fosc().run_stda

[Show source in Objective_function.py:356](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L356)

#### Signature

```python
def run_stda(
    self,
    polymer,
    STDA_bin_path,
    output_dir,
    property="Excited state energy (eV)",
    state=1,
    database="stk_mohammed",
    collection="test",
    client=None,
): ...
```

### IP_ES1_fosc().run_xtb_ipea

[Show source in Objective_function.py:310](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L310)

#### Signature

```python
def run_xtb_ipea(
    self,
    polymer,
    xtb_path,
    xtb_opt_output_dir,
    database="stk_mohammed_BO",
    collection="testIPEA",
    target="ionisation potential (eV)",
    client=None,
): ...
```

### IP_ES1_fosc().run_xtb_opt

[Show source in Objective_function.py:189](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L189)

#### Signature

```python
def run_xtb_opt(
    self,
    polymer,
    xtb_path,
    xtb_opt_output_dir,
    database="stk_mohammed_BO",
    collection="test",
    client=None,
): ...
```



## Look_up_table

[Show source in Objective_function.py:28](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L28)

#### Signature

```python
class Look_up_table:
    def __init__(self, df_look_up, fragment_size, target_name="target", aim=0): ...
```

### Look_up_table().evaluate_element

[Show source in Objective_function.py:35](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L35)

#### Signature

```python
def evaluate_element(self, element, multiFidelity=False): ...
```



## Objective_Function

[Show source in Objective_function.py:18](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L18)

#### Signature

```python
class Objective_Function:
    def __init__(self): ...
```

### Objective_Function().evaluate_element

[Show source in Objective_function.py:22](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L22)

#### Signature

```python
def evaluate_element(self, element, multiFidelity=False): ...
```



## get_inchi_key

[Show source in Objective_function.py:14](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/Objective_function.py#L14)

#### Signature

```python
def get_inchi_key(molecule): ...
```