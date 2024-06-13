# Introduction 
Code for ICML 2024 paper "Efficient and Effective Time-Series Forecasting with Spiking Neural Networks"(https://arxiv.org/pdf/2402.01533)

# Installation
`python setup.py install`

## Training
Take the `iSpikformer` model as an example:

`python tsforecast.py exp/forecast/ispikformer_electricity.yml`

You can change the `yml` configuration files as you want.

We also provide several classic ANN models for time-series forecasting, such as TCN, Autoformer, LSTNet, and MTGNN.
You can add, remove, or modify your model architecture in `forecaster/network/XXX.py`.

## Datasets

Metr-la and Pems-bay are available at at [Google Drive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g).
Solar and Electricity can be downloaded from  (https://github.com/laiguokun/multivariate-time-series-data).

The folder structure of this procect is as follows:
```
SeqSNN
│   README.md 
│   ...
│
└───data
│   │   metr-la.h5
│   │   pems-bay.h5
│   │
│   └───solar-energy
│   │   │   solar_AL.txt
│   │   │   ...
│   │   
│   └───electricity
│   │   │   electricity.txt
│   │   │   ...
│   │   
│   └───traffic
│   │   │   traffic.txt
│   │   │   ...
│
└───forecaster
│   │   ...
│
└───exp
│   │   ...
│
└───outputs
│   │   ...
│
```
You can change the path of data file in `exp/forecast/dataset/XXX.yml` configuration files.
