# Searchexp

[Stk_search Index](../../README.md#stk_search-index) / `src` / [Stk Search](./index.md#stk-search) / Searchexp

> Auto-generated documentation for [src.stk_search.SearchExp](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchExp.py) module.

- [Searchexp](#searchexp)
  - [SearchExp](#searchexp)
    - [SearchExp().evaluate_element](#searchexp()evaluate_element)
    - [SearchExp().run_seach](#searchexp()run_seach)
    - [SearchExp().save_results](#searchexp()save_results)
    - [SearchExp().save_search_experiment](#searchexp()save_search_experiment)

## SearchExp

[Show source in SearchExp.py:12](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchExp.py#L12)

#### Signature

```python
class SearchExp:
    def __init__(
        self,
        searchspace: SearchSpace,
        search_algorithm,
        objective_function,
        number_of_iterations,
        verbose=False,
    ): ...
```

### SearchExp().evaluate_element

[Show source in SearchExp.py:114](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchExp.py#L114)

#### Signature

```python
def evaluate_element(
    self, element_id: int, objective_function: Objective_Function = None
): ...
```

### SearchExp().run_seach

[Show source in SearchExp.py:44](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchExp.py#L44)

#### Signature

```python
def run_seach(self): ...
```

### SearchExp().save_results

[Show source in SearchExp.py:162](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchExp.py#L162)

#### Signature

```python
def save_results(self): ...
```

### SearchExp().save_search_experiment

[Show source in SearchExp.py:149](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/SearchExp.py#L149)

#### Signature

```python
def save_search_experiment(self): ...
```