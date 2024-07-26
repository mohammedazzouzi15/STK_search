# Searchedspace

[Stk_search Index](../../README.md#stk_search-index) / `src` / [Stk Search](./index.md#stk-search) / Searchedspace

> Auto-generated documentation for [src.stk_search.SearchedSpace](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchedSpace.py) module.

- [Searchedspace](#searchedspace)
  - [SearchedSpace](#searchedspace)
    - [SearchedSpace().generate_interactive_condition_V2](#searchedspace()generate_interactive_condition_v2)
    - [SearchedSpace().get_all_possible_syntax](#searchedspace()get_all_possible_syntax)
    - [SearchedSpace().plot_hist_compare](#searchedspace()plot_hist_compare)
    - [SearchedSpace().plot_histogram_fragment](#searchedspace()plot_histogram_fragment)

## SearchedSpace

[Show source in SearchedSpace.py:16](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchedSpace.py#L16)

#### Signature

```python
class SearchedSpace(SearchSpace): ...
```

### SearchedSpace().generate_interactive_condition_V2

[Show source in SearchedSpace.py:144](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchedSpace.py#L144)

#### Signature

```python
def generate_interactive_condition_V2(
    self, df_total: pd.DataFrame, properties_to_plot=[]
): ...
```

### SearchedSpace().get_all_possible_syntax

[Show source in SearchedSpace.py:115](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchedSpace.py#L115)

#### Signature

```python
def get_all_possible_syntax(self): ...
```

### SearchedSpace().plot_hist_compare

[Show source in SearchedSpace.py:17](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchedSpace.py#L17)

#### Signature

```python
def plot_hist_compare(self, df_all, df_list, label_list, properties_to_plot=[]): ...
```

### SearchedSpace().plot_histogram_fragment

[Show source in SearchedSpace.py:57](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchedSpace.py#L57)

#### Signature

```python
def plot_histogram_fragment(
    self, column_name, df_list, df_total, number_of_fragments, label_list
): ...
```