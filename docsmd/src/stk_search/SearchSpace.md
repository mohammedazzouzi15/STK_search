# Searchspace

[stk_search Index](../../README.md#stk_search-index) / `src` / [Stk Search](./index.md#stk-search) / Searchspace

> Auto-generated documentation for [src.stk_search.SearchSpace](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py) module.

- [Searchspace](#searchspace)
  - [SearchSpace](#searchspace)
    - [SearchSpace().add_condition](#searchspace()add_condition)
- [condition syntax should follow the following condition:](#condition-syntax-should-follow-the-following-condition:)
- ["'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"](#"'column'#operation#value"-eg-"'ip-(ev)'#>=#65")
    - [SearchSpace().check_df_for_element_from_SP](#searchspace()check_df_for_element_from_sp)
    - [SearchSpace().generate_dataframe_with_search_space](#searchspace()generate_dataframe_with_search_space)
    - [SearchSpace().generate_list_fragment](#searchspace()generate_list_fragment)
    - [SearchSpace().generate_syntax](#searchspace()generate_syntax)
    - [SearchSpace().get_space_size](#searchspace()get_space_size)
    - [SearchSpace().plot_histogram_precursor](#searchspace()plot_histogram_precursor)
    - [SearchSpace().remove_condition](#searchspace()remove_condition)
    - [SearchSpace().update](#searchspace()update)

## SearchSpace

[Show source in SearchSpace.py:10](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L10)

class that contains the chemical space to search over
it is defined by the number of fragments and the syntax of the fragment forming the oligomer
it also contains the conditions that need to be respected by the building blocks

Attributes
----------
number_of_fragments : int
    number of fragments in the oligomer
df_precursors : pd.DataFrame
    dataframe containing the building blocks inchikeys and features
generation_type : str
    type of generation of the search space
syntax : list
    list of the syntax of the oligomer
conditions_list : list
    list of the conditions that need to be respected by the building blocks

#### Signature

```python
class SearchSpace:
    def __init__(
        self,
        number_of_fragments: int,
        df: pd.DataFrame,
        generation_type: str = "conditional",
    ): ...
```

### SearchSpace().add_condition

[Show source in SearchSpace.py:59](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L59)

add a condition to the condition list
# condition syntax should follow the following condition:
# "'column'#operation#value" e.g. "'IP (eV)'#>=#6.5"
Parameters
----------
condition : str
    condition to add
fragment : int
    fragment position to which the condition is added

#### Signature

```python
def add_condition(self, condition: str, fragment: int): ...
```

### SearchSpace().check_df_for_element_from_SP

[Show source in SearchSpace.py:80](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L80)

#### Signature

```python
def check_df_for_element_from_SP(self, df_to_check: pd.DataFrame): ...
```

### SearchSpace().generate_dataframe_with_search_space

[Show source in SearchSpace.py:174](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L174)

#### Signature

```python
def generate_dataframe_with_search_space(self): ...
```

### SearchSpace().generate_list_fragment

[Show source in SearchSpace.py:124](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L124)

#### Signature

```python
def generate_list_fragment(self): ...
```

### SearchSpace().generate_syntax

[Show source in SearchSpace.py:167](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L167)

#### Signature

```python
def generate_syntax(self): ...
```

### SearchSpace().get_space_size

[Show source in SearchSpace.py:154](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L154)

#### Signature

```python
def get_space_size(self): ...
```

### SearchSpace().plot_histogram_precursor

[Show source in SearchSpace.py:221](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L221)

#### Signature

```python
def plot_histogram_precursor(self): ...
```

### SearchSpace().remove_condition

[Show source in SearchSpace.py:75](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L75)

#### Signature

```python
def remove_condition(self, condition: str, fragment: int): ...
```

### SearchSpace().update

[Show source in SearchSpace.py:114](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchSpace.py#L114)

update the search space based on the conditions
changes the list of fragment and recomputes the space size

#### Signature

```python
def update(self): ...
```