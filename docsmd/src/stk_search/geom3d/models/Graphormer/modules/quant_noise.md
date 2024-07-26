# Quant Noise

[stk_search Index](../../../../../../README.md#stk_search-index) / `src` / [Stk Search](../../../../index.md#stk-search) / [Stk Search](../../../../index.md#stk-search) / [Models](../../index.md#models) / [Graphormer](../index.md#graphormer) / [Modules](./index.md#modules) / Quant Noise

> Auto-generated documentation for [src.stk_search.geom3d.models.Graphormer.modules.quant_noise](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/quant_noise.py) module.

- [Quant Noise](#quant-noise)
  - [quant_noise](#quant_noise)

## quant_noise

[Show source in quant_noise.py:6](https://github.com/mohammedazzouzi15/STK_search/blob/main/src/stk_search/geom3d/models/Graphormer/modules/quant_noise.py#L6)

Wraps modules and applies quantization noise to the weights for
subsequent quantization with Iterative Product Quantization as
described in "Training with Quantization Noise for Extreme Model Compression"

#### Arguments

    - `-` *module* - nn.Module
    - `-` *p* - amount of Quantization Noise
    - `-` *block_size* - size of the blocks for subsequent quantization with iPQ
Remarks:
    - Module weights must have the right sizes wrt the block size
    - Only Linear, Embedding and Conv2d modules are supported for the moment
    - For more detail on how to quantize by blocks with convolutional weights,
      see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
    - We implement the simplest form of noise here as stated in the paper
      which consists in randomly dropping blocks

#### Signature

```python
def quant_noise(module, p, block_size): ...
```