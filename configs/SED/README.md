# SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation


[SED: A Simple Encoder-Decoder for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2311.15537)

<!-- [ALGORITHM] -->

## Abstract

SED is an open-vocabulary semantic segmentation model that features a hierarchical encoder-based cost map generation and a gradual fusion decoder with an early category rejection mechanism. Unlike plain transformers, the hierarchical backbone more effectively captures local spatial information and maintains linear computational complexity relative to input size. The early category rejection scheme in the decoder discards many non-existent categories in the early stages of decoding, significantly speeding up inference by up to 4.7 times without sacrificing accuracy.


![](./assets/Overview.png)
 
## Installation
We install SED using the official [github repository](https://github.com/xb534/SED) and follow the [instructions](https://github.com/xb534/SED/blob/main/INSTALL.md) to configure the environment.

