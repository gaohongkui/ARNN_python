<!--
 * @Date: 2021-08-10 10:55:32
 * @LastEditors: GodK
-->
# ARNN_Python

##  Introduction

Implement [Autoreservoir computing for multistep ahead prediction based on the spatiotemporal information transformation](https://www.nature.com/articles/s41467-020-18381-0) with Python. The code refers to the matlab program of the original author's [repo](https://github.com/RPcb/ARNN).


## Usage

First, install the necessary dependent packages:

```bash
pip install -r requirements.txt
```

For Lorenz model simulation, there are the following three cases:

* noise-free & time-invariant case: use `mylorenz.py` to generate high-dimensional data, set `noisestrength = 0` in `main.py`;

* noisy & time-invariant case: use `mylorenz.py` to generate high-dimensional data, set `noisestrength` to be `0.1-1.0` in `main.py`, respectively;

* time-varying case: use `mylorenz_dynamic.m` to generate high-dimensional data, set `noisestrength = 0` in `main.py`.

## Demo

The code `LongerPredictionSamples_ARNN.py` in repository can generates the results in Figure 2d,2e,2f of the main text.

Expected running time for this demo is less than 1 minute on a "normal" desktop computer.